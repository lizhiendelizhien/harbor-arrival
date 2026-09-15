"""Shared MLP latent, causal dynamic decoder, and FSQ reconstruction branch."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from vector_quantize_pytorch import FSQ

from gear_sonic.finance.denoising import prepare_encoder_input
from gear_sonic.finance.observations import CURRENT_FIELDS, FUTURE_FIELDS
from gear_sonic.trl.modules.base_module import BaseModule
from gear_sonic.trl.modules.transformer_decoder import TransformerActionDecoder


@dataclass(frozen=True)
class FinancialSonicConfig:
    horizon: int = 10
    mlp_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 512)
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 6
    ffn_dim: int = 1024
    window_size: int = 32
    dropout: float = 0.0
    normalization_clip: float = 10.0


class RobustNormalizer(nn.Module):
    """Reference-pool median/IQR statistics persisted with the model checkpoint."""

    def __init__(self, features: int, clip: float):
        super().__init__()
        self.clip = clip
        self.register_buffer("center", torch.zeros(features))
        self.register_buffer("scale", torch.ones(features))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, values: torch.Tensor, mask: torch.Tensor | None = None):
        if values.shape[-1] != self.center.numel():
            raise ValueError("Unexpected normalizer feature dimension")
        if mask is None:
            mask = torch.ones_like(values, dtype=torch.bool)
        if mask.shape != values.shape or mask.dtype != torch.bool:
            raise ValueError("Normalizer mask must be boolean with the same shape as values")
        if not torch.isfinite(values[mask]).all():
            raise ValueError("Normalizer training values must be finite where valid")
        flat = torch.where(mask, values, torch.nan).reshape(-1, values.shape[-1]).float()
        if not torch.isfinite(flat).any(dim=0).all():
            raise ValueError("Every normalizer feature needs valid training values")
        quantiles = torch.nanquantile(flat, flat.new_tensor([0.25, 0.5, 0.75]), dim=0)
        spread = quantiles[2] - quantiles[0]
        self.center.copy_(quantiles[1])
        self.scale.copy_(torch.where(spread > 1e-6, spread, torch.ones_like(spread)))
        self.fitted.fill_(True)

    def forward(self, values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if not bool(self.fitted):
            raise RuntimeError("Fit normalizers on training data or load a fitted checkpoint first")
        if mask is None:
            mask = torch.ones_like(values, dtype=torch.bool)
        if not torch.isfinite(values[mask]).all():
            raise ValueError("Observations must be finite where valid")
        clean = torch.where(mask, values, self.center)
        scaled = ((clean - self.center) / self.scale).clamp(-self.clip, self.clip)
        return torch.where(mask, scaled, 0.0)

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.scale + self.center


class FinancialSonic(nn.Module):
    """Privileged trajectory reconstruction with an external-latent inference API.

    B is batch, S is successive monthly anchors, H is the future horizon.
    Forward: current [B,S,16], future [B,S,H,15]. Each anchor has its own
    future window. Streaming: current [B,16], externally supplied raw z [B,64].
    """

    def __init__(self, config: FinancialSonicConfig | None = None):
        super().__init__()
        self.config = config or FinancialSonicConfig()
        cfg = self.config
        if cfg.horizon < 1 or cfg.num_layers < 1 or cfg.normalization_clip <= 0:
            raise ValueError("Horizon, layer count and normalization clip must be positive")
        if not cfg.mlp_hidden_dims or any(dim < 1 for dim in cfg.mlp_hidden_dims):
            raise ValueError("MLP hidden dimensions must be positive")
        mlp_config = SimpleNamespace(layer_config={
            "type": "MLP", "hidden_dims": list(cfg.mlp_hidden_dims), "activation": "SiLU",
        })
        self.encoder = BaseModule(
            input_dim=len(FUTURE_FIELDS), output_dim=32, num_input_temporal_dims=cfg.horizon,
            num_output_temporal_dims=2, module_config_dict=mlp_config,
        )
        # Scalar codes are sufficient; a joint 32**32 codebook is not enumerated.
        self.quantizer = FSQ(levels=[32] * 32, return_indices=False)
        self.kin = BaseModule(
            input_dim=32, output_dim=len(FUTURE_FIELDS), num_input_temporal_dims=2,
            num_output_temporal_dims=cfg.horizon, module_config_dict=mlp_config,
        )
        self.dyn = TransformerActionDecoder(
            input_dim=len(CURRENT_FIELDS) + 64, output_dim=cfg.horizon,
            d_model=cfg.d_model, num_heads=cfg.num_heads, num_layers=cfg.num_layers,
            ffn_dim=cfg.ffn_dim, window_size=cfg.window_size, dropout=cfg.dropout,
            attention_dropout=cfg.dropout,
        )
        self.current_normalizer = RobustNormalizer(len(CURRENT_FIELDS), cfg.normalization_clip)
        self.future_normalizer = RobustNormalizer(len(FUTURE_FIELDS), cfg.normalization_clip)

    def fit_normalizers(self, current: torch.Tensor, future: torch.Tensor, future_mask=None):
        """Fit the chosen reference pool; forward never updates statistics."""
        self.current_normalizer.fit(current)
        self.future_normalizer.fit(future, future_mask)
        self.reset_cache()

    @torch.no_grad()
    def load_normalizers(self, path: str | Path):
        """Load exported pool statistics; model inputs remain in original units."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported normalization schema")
        if payload.get("horizon") != self.config.horizon:
            raise ValueError("Normalization horizon must match model horizon")
        validated = []
        for name, fields, normalizer in (
            ("current", CURRENT_FIELDS, self.current_normalizer),
            ("future", FUTURE_FIELDS, self.future_normalizer),
        ):
            stats = payload[name]
            if stats["fields"] != list(fields):
                raise ValueError(f"Unexpected {name} feature schema")
            center = normalizer.center.new_tensor(stats["center"])
            scale = normalizer.scale.new_tensor(stats["scale"])
            if (center.shape != normalizer.center.shape or scale.shape != normalizer.scale.shape
                    or not torch.isfinite(center).all() or not torch.isfinite(scale).all()
                    or not (scale > 0).all() or stats["clip"] != normalizer.clip):
                raise ValueError(f"Invalid {name} normalization statistics or clip")
            validated.append((normalizer, center, scale))
        for normalizer, center, scale in validated:
            normalizer.center.copy_(center)
            normalizer.scale.copy_(scale)
            normalizer.fitted.fill_(True)
        self.reset_cache()

    def _future_mask(self, future: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if future.ndim < 3 or future.shape[-2:] != (self.config.horizon, len(FUTURE_FIELDS)):
            raise ValueError(f"Future observations must end in [{self.config.horizon},15]")
        if mask is None:
            mask = torch.ones_like(future, dtype=torch.bool)
        if mask.shape != future.shape or mask.dtype != torch.bool:
            raise ValueError("future_mask must be boolean with shape equal to future observations")
        if not mask.any(dim=(-1, -2)).all():
            raise ValueError("Every future window needs at least one valid feature")
        return mask

    def encode(
        self, future: torch.Tensor, future_mask: torch.Tensor | None = None, *,
        encoder_noise: torch.Tensor | None = None, encoder_mask_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode clean or explicitly corrupted future data to unquantized raw z."""
        mask = self._future_mask(future, future_mask)
        inputs = prepare_encoder_input(
            self.future_normalizer(future, mask), mask,
            encoder_noise=encoder_noise, encoder_mask_type=encoder_mask_type,
        )
        return self.encoder(inputs).flatten(-2)

    def _trajectory(self, normalized_returns: torch.Tensor) -> dict[str, torch.Tensor]:
        returns = normalized_returns * self.future_normalizer.scale[0] + self.future_normalizer.center[0]
        return {"normalized_actions": normalized_returns, "log_returns": returns,
                "cumulative_log_returns": returns.cumsum(dim=-1)}

    def decode(
        self, current: torch.Tensor, latent: torch.Tensor, episode_attnmask=None,
        *, kv_prefix_state=None, kv_prefix_dones=None,
    ) -> dict:
        """Decode successive current states and raw latents with causal attention."""
        if current.ndim != 3 or current.shape[-1] != len(CURRENT_FIELDS):
            raise ValueError("Current states must have shape [B,S,16]")
        if latent.shape != (*current.shape[:2], 64) or not torch.isfinite(latent).all():
            raise ValueError("Raw latent must be finite with shape [B,S,64]")
        inputs = torch.cat((self.current_normalizer(current), latent), dim=-1)
        return self._trajectory(self.dyn(
            inputs, episode_attnmask=episode_attnmask,
            kv_prefix_state=kv_prefix_state, kv_prefix_dones=kv_prefix_dones,
        ))

    def forward(
        self, current: torch.Tensor, future: torch.Tensor, *, future_mask=None, episode_attnmask=None,
        kv_prefix_state=None, kv_prefix_dones=None, encoder_noise=None, encoder_mask_type=None,
    ) -> dict:
        if future.ndim != 4 or current.shape[:2] != future.shape[:2]:
            raise ValueError("Expected matching [B,S,16] current and [B,S,H,15] future observations")
        mask = self._future_mask(future, future_mask)
        normalized_future = self.future_normalizer(future, mask)
        encoder_input = prepare_encoder_input(
            normalized_future, mask, encoder_noise=encoder_noise, encoder_mask_type=encoder_mask_type,
        )
        tokens = self.encoder(encoder_input)
        latent = tokens.flatten(-2)
        quantized, _ = self.quantizer(tokens.reshape(-1, 2, 32))
        quantized = quantized.reshape_as(tokens)
        reconstruction = self.kin(quantized)
        # Re-encode clean normalized reconstructions with the shared encoder.
        reencoded = self.encoder(reconstruction).flatten(-2)
        result = self.decode(current, latent, episode_attnmask,
                             kv_prefix_state=kv_prefix_state, kv_prefix_dones=kv_prefix_dones)
        result.update({
            "latent": latent, "quantized_latent": quantized,
            "kin_reconstruction": self.future_normalizer.inverse(reconstruction),
            "kin_normalized": reconstruction, "future_normalized": normalized_future,
            "reencoded_latent": reencoded, "future_mask": mask,
        })
        return result

    @torch.no_grad()
    def predict_step(self, current: torch.Tensor, latent: torch.Tensor, *, reset_mask=None) -> dict:
        """Advance cache once per month. Call eval() and reset on symbol/episode changes.

        This API requires an external raw latent; it never accesses future data.
        """
        if self.training:
            raise RuntimeError("Call eval() before cached inference")
        if current.ndim != 2 or current.shape[-1] != len(CURRENT_FIELDS):
            raise ValueError("Streaming current states must have shape [B,16]")
        if latent.shape != (current.shape[0], 64) or not torch.isfinite(latent).all():
            raise ValueError("Streaming latent must be finite with shape [B,64]")
        inputs = torch.cat((self.current_normalizer(current), latent), dim=-1)
        return self._trajectory(self.dyn.forward_step(inputs, reset_mask=reset_mask)[:, 0])

    def reset_cache(self, reset_mask: torch.Tensor | None = None):
        self.dyn.reset_cache(reset_mask)
