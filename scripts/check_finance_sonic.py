"""Run a bounded architecture check on one reference pool, without dataset splits.

All supplied dates are included by default. Optional date bounds focus normalizer
fitting or playback checks; the result measures reference reconstruction only.

Usage: python -m scripts.check_finance_sonic --canonical PATH --symbols AAPL MSFT
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from gear_sonic.finance.losses import financial_sonic_loss
from gear_sonic.finance.model import FinancialSonic, FinancialSonicConfig
from gear_sonic.finance.observations import iter_sequences, load_observations


def _tensors(sequences):
    return (
        torch.tensor([row["current_state"] for row in sequences], dtype=torch.float32),
        torch.tensor([row["future_reference"] for row in sequences], dtype=torch.float32),
        torch.tensor([row["future_mask"] for row in sequences], dtype=torch.bool),
    )


def run_check(
    canonical: str | Path, *, symbols: set[str], sequence_length: int = 32,
    fit_start: str | None = None, fit_end: str | None = None,
    check_start: str | None = None, check_end: str | None = None, tiny: bool = False,
) -> dict:
    observations = load_observations(canonical, symbols)
    fit_sequences = list(iter_sequences(
        observations, sequence_length=sequence_length, start_period=fit_start, end_period=fit_end,
    ))
    check_sequences = []
    for symbol, rows in observations.items():
        sequence = next(iter_sequences(
            {symbol: rows}, sequence_length=sequence_length,
            start_period=check_start, end_period=check_end,
        ), None)
        if sequence is not None:
            check_sequences.append(sequence)
        if len(check_sequences) == 2:
            break
    if not fit_sequences or not check_sequences:
        raise ValueError("Not enough contiguous observations for both fit and check sequences")
    fit_current, fit_future, fit_mask = _tensors(fit_sequences)
    current, future, mask = _tensors(check_sequences)
    config = FinancialSonicConfig()
    if tiny:
        config = FinancialSonicConfig(mlp_hidden_dims=(32, 32), d_model=32,
                                      num_heads=2, num_layers=2, ffn_dim=64)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            model = FinancialSonic(config)
        model.fit_normalizers(fit_current, fit_future, fit_mask)
        # One backward pass on the reference pool checks gradients without updating weights.
        training_output = model(fit_current[:1], fit_future[:1], future_mask=fit_mask[:1])
        losses = financial_sonic_loss(
            training_output, fit_future[:1], return_scale=model.future_normalizer.scale[0],
        )
        losses["total"].backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError("Nonfinite gradient in real-data check")
        encoder_norm = sum(p.grad.square().sum() for p in model.encoder.parameters() if p.grad is not None).sqrt()
        if not encoder_norm > 0:
            raise RuntimeError("No encoder gradient in real-data check")
        model.eval()
        with torch.no_grad():
            output = model(current, future, future_mask=mask)
            model.reset_cache()
            streamed = torch.stack([
                model.predict_step(current[:, index], output["latent"][:, index])["log_returns"]
                for index in range(sequence_length)
            ], dim=1)
            torch.testing.assert_close(streamed, output["log_returns"], atol=1e-5, rtol=1e-5)
            if not all(torch.isfinite(value).all() for value in output.values()):
                raise RuntimeError("Nonfinite output in real-data check")
        return {
            "mode": "privileged_reconstruction_smoke_check",
            "dataset_partition": "none",
            "config": asdict(config),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "fit_sequence_count": len(fit_sequences),
            "fit_period": [fit_start, fit_end], "check_period": [check_start, check_end],
            "check_sequences": [{"symbol": row["symbol"], "anchor_start": row["periods"][0],
                                 "anchor_end": row["periods"][-1], "target_end": row["target_end_period"]}
                                for row in check_sequences],
            "input_shapes": {"current": list(current.shape), "future": list(future.shape)},
            "output_shapes": {key: list(output[key].shape) for key in
                              ("latent", "quantized_latent", "kin_reconstruction", "log_returns")},
            "fit_batch_losses": {key: float(value.detach()) for key, value in losses.items()},
            "encoder_gradient_norm": float(encoder_norm),
            "streaming_max_abs_error": float((streamed - output["log_returns"]).abs().max()),
        }
    finally:
        torch.set_num_threads(previous_threads)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT"])
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--fit-start", help="Optional first month for normalizer fitting")
    parser.add_argument("--fit-end", help="Optional last month for normalizer fitting")
    parser.add_argument("--check-start", help="Optional first month for playback checks")
    parser.add_argument("--check-end", help="Optional last month for playback checks")
    parser.add_argument("--tiny", action="store_true", help="Use a small model for a fast CPU check")
    args = parser.parse_args(argv)
    result = run_check(
        args.canonical, symbols=set(args.symbols), sequence_length=args.sequence_length,
        fit_start=args.fit_start, fit_end=args.fit_end, check_start=args.check_start,
        check_end=args.check_end, tiny=args.tiny,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
