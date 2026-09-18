"""Rank-aware TensorBoard monitoring for Finance Sonic training."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path
from typing import Any

import torch


_LOSS_NAMES = {"loss", "ppo_loss", "policy_loss", "value_loss", "kin", "cycle"}
_POLICY_NAMES = {"entropy", "kl", "grad_norm", "learning_rate"}
_TRAIN_NAMES = {"reward", "sample_count", "terminal_count", "updates"}


def tensorboard_tag(name: str) -> str | None:
    """Map a trainer metric name to the table-tennis-style scalar namespace."""
    if name in {"iteration", "post_update_checksum"}:
        return None
    if name in _LOSS_NAMES:
        return f"Loss/{name}"
    if name in _POLICY_NAMES:
        return f"Policy/{name}"
    if name in _TRAIN_NAMES:
        return f"Train/{name}"
    if name.startswith(("reward_", "contribution_")):
        return f"Reward/{name}"
    return f"Tracking/{name}"


def _finite_scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    if not isinstance(value, Real) or isinstance(value, bool):
        return None
    scalar = float(value)
    return scalar if math.isfinite(scalar) else None


@dataclass
class FinanceTensorBoardMonitor:
    """Small writer wrapper that is safe to instantiate on every rank."""

    writer: Any = None
    enabled: bool = False
    _closed: bool = False

    def log(self, metrics: dict[str, Any]) -> None:
        if not self.enabled or self.writer is None or self._closed:
            return
        iteration = metrics.get("iteration")
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 1:
            raise ValueError("TensorBoard metrics require a positive integer iteration")
        for name, value in metrics.items():
            tag = tensorboard_tag(str(name))
            if tag is None:
                continue
            scalar = _finite_scalar(value)
            if scalar is not None:
                self.writer.add_scalar(tag, scalar, iteration)

    def flush(self) -> None:
        if self.enabled and self.writer is not None and not self._closed:
            self.writer.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self.enabled or self.writer is None:
            return
        try:
            self.writer.flush()
        finally:
            self.writer.close()


def create_tensorboard_monitor(
    output_dir: Path,
    *,
    enabled: bool,
    is_main_process: bool,
    resume_iteration: int = 0,
) -> FinanceTensorBoardMonitor:
    """Create the rank-zero writer without importing TensorBoard elsewhere."""
    if not enabled or not is_main_process:
        return FinanceTensorBoardMonitor()
    if resume_iteration < 0:
        raise ValueError("resume_iteration must be non-negative")

    event_dir = Path(output_dir) / "tensorboard"
    event_dir.mkdir(parents=True, exist_ok=True)
    has_existing_events = any(event_dir.glob("events.out.tfevents.*"))
    writer_kwargs = {"log_dir": str(event_dir), "flush_secs": 10}
    if has_existing_events and resume_iteration > 0:
        writer_kwargs["purge_step"] = resume_iteration + 1
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(**writer_kwargs)
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard logging is enabled but torch.utils.tensorboard is unavailable. "
            "Install it with $PYTHON_BIN -m pip install -e '.[finance]' or set LOGGER=none."
        ) from exc
    return FinanceTensorBoardMonitor(writer=writer, enabled=True)
