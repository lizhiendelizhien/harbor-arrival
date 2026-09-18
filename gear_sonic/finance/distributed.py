"""Small process-group helpers for the finance PPO trainer.

The finance Actor owns a stateful KV cache and exposes methods beyond
``forward``.  Consequently the trainer uses explicit gradient all-reduce
instead of wrapping the whole Actor in :class:`DistributedDataParallel`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os

import torch
import torch.distributed as dist


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if parsed < 0:
        raise ValueError(f"{name} must be nonnegative")
    return parsed


@dataclass
class DistributedContext:
    """Runtime state and collectives used by one finance training rank."""

    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str | None = None
    owns_process_group: bool = False
    device: torch.device | None = None
    _rms_snapshots: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = field(
        default_factory=dict, init=False, repr=False,
    )

    @classmethod
    def initialize(cls, *, requested: bool = False, device=None, backend: str | None = None):
        """Initialize ``env://`` process-group state when torchrun requested it."""
        world_size = _env_int("WORLD_SIZE", 1)
        if not requested and world_size <= 1:
            return cls(device=torch.device(device) if device is not None else None)
        if world_size <= 1:
            # ``--distributed`` is also useful in unit tests without torchrun;
            # keep that invocation a harmless single-process run.
            return cls(device=torch.device(device) if device is not None else None)
        if not dist.is_available():
            raise RuntimeError("This PyTorch build does not provide torch.distributed")
        rank = _env_int("RANK", 0)
        local_rank = _env_int("LOCAL_RANK", rank)
        if rank >= world_size:
            raise ValueError(f"RANK={rank} is out of range for WORLD_SIZE={world_size}")
        if dist.is_initialized():
            active_backend = dist.get_backend()
            active_world = dist.get_world_size()
            active_rank = dist.get_rank()
            if active_world != world_size or active_rank != rank:
                raise ValueError("Existing process group disagrees with torchrun environment")
            return cls(True, rank, local_rank, world_size, active_backend, False,
                       torch.device(device) if device is not None else None)

        if backend is None:
            backend = os.environ.get("FINANCE_DDP_BACKEND")
        if backend is None:
            device_type = torch.device(device).type if device is not None else "cpu"
            backend = "nccl" if device_type == "cuda" else "gloo"
        if backend == "nccl" and not torch.cuda.is_available():
            raise RuntimeError("NCCL DDP requested but CUDA is unavailable")
        dist.init_process_group(backend=backend, init_method="env://")
        return cls(True, rank, local_rank, world_size, backend, True,
                   torch.device(device) if device is not None else None)

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def all_reduce_(self, value: torch.Tensor, *, op=dist.ReduceOp.SUM) -> torch.Tensor:
        if self.enabled:
            dist.all_reduce(value, op=op)
        return value

    def _collective_value(self, value, device=None) -> torch.Tensor:
        target = device if device is not None else self.device
        if isinstance(value, torch.Tensor):
            result = value.detach().clone()
            if target is not None and result.device != torch.device(target):
                result = result.to(target)
            return result
        return torch.as_tensor(value, device=target).detach().clone()

    def mean(self, value: torch.Tensor | float, *, device=None) -> torch.Tensor:
        result = self._collective_value(value, device)
        if not result.is_floating_point() and not result.is_complex():
            result = result.float()
        if self.enabled:
            self.all_reduce_(result)
            result /= self.world_size
        return result

    def sum(self, value: torch.Tensor | float, *, device=None) -> torch.Tensor:
        result = self._collective_value(value, device)
        if self.enabled:
            self.all_reduce_(result)
        return result

    def standardize(self, values: torch.Tensor) -> torch.Tensor:
        """Normalize with moments pooled across ranks, preserving the shape."""
        if values.numel() == 0:
            return values
        flat = values.detach() if not values.requires_grad else values
        count = flat.new_tensor(float(flat.numel()))
        total = flat.sum()
        squared = flat.square().sum()
        if self.enabled:
            self.all_reduce_(count)
            self.all_reduce_(total)
            self.all_reduce_(squared)
        count_safe = count.clamp_min(1.0)
        mean = total / count_safe
        centered_sum = (squared - total * mean).clamp_min(0.0)
        variance = centered_sum / (count - 1.0).clamp_min(1.0)
        return (values - mean) / (variance.sqrt() + 1e-8)

    def sync_gradients(self, parameters) -> None:
        """Average gradients with coalesced collectives, including unused params."""
        if not self.enabled:
            return
        groups = {}
        for parameter in parameters:
            if parameter.numel() == 0:
                continue
            if parameter.grad is not None and parameter.grad.is_sparse:
                raise RuntimeError("Finance DDP does not support sparse parameter gradients")
            key = (parameter.device.type, parameter.device.index, parameter.dtype)
            groups.setdefault(key, []).append(parameter)

        # A single flat buffer per device/dtype keeps the explicit reduction
        # compatible with the stateful KV-cache Actor while avoiding one
        # collective call per parameter tensor.
        for group in groups.values():
            parts = []
            for parameter in group:
                if parameter.grad is None:
                    gradient = torch.zeros_like(parameter)
                else:
                    gradient = parameter.grad.detach()
                    if gradient.dtype != parameter.dtype:
                        gradient = gradient.to(dtype=parameter.dtype)
                parts.append(gradient.reshape(-1))
            flat = torch.cat(parts, dim=0)
            self.all_reduce_(flat)
            flat /= self.world_size
            offset = 0
            for parameter in group:
                size = parameter.numel()
                reduced = flat.narrow(0, offset, size).view_as(parameter)
                if parameter.grad is None:
                    parameter.grad = reduced.clone()
                else:
                    parameter.grad.copy_(reduced)
                offset += size

    @torch.no_grad()
    def broadcast_module(self, module: torch.nn.Module, *, source_rank: int = 0) -> None:
        """Match DDP initialization by broadcasting parameters and buffers."""
        if not self.enabled:
            return
        if not 0 <= source_rank < self.world_size:
            raise ValueError("source_rank is outside the distributed world")
        for parameter in module.parameters():
            dist.broadcast(parameter, src=source_rank)
        for buffer in module.buffers():
            dist.broadcast(buffer, src=source_rank)

    def sync_running_mean_std(self, normalizer) -> None:
        """Pool Critic running moments after each local observation batch."""
        if not self.enabled or normalizer is None:
            return
        with torch.no_grad():
            current_count = normalizer.count.detach().clone().to(dtype=torch.float64)
            current_mean = normalizer.running_mean.detach().clone().to(dtype=torch.float64)
            current_var = normalizer.running_var.detach().clone().to(dtype=torch.float64)
            key = id(normalizer)
            previous = self._rms_snapshots.get(key)
            if previous is None:
                # RunningMeanStd is replicated on every rank at startup and on
                # resume.  Average those copies so the prior pseudo-count is
                # not multiplied by world_size.
                count = current_count
                mean = current_mean
                second = current_var + current_mean.square()
                self.all_reduce_(count)
                self.all_reduce_(mean)
                self.all_reduce_(second)
                count /= self.world_size
                mean /= self.world_size
                second /= self.world_size
                variance = (second - mean.square()).clamp_min(1e-12)
            else:
                old_count, old_mean, old_var = previous
                delta_count = (current_count - old_count).clamp_min(0.0)
                batch_total = torch.zeros_like(current_mean)
                batch_second = torch.zeros_like(current_mean)
                has_batch = bool(delta_count.item() > 0)
                if has_batch:
                    # Invert RunningMeanStd's parallel-moments update to
                    # recover the local batch, then combine batches globally.
                    total_count = current_count.clamp_min(1.0)
                    batch_mean = (current_mean * total_count - old_mean * old_count) / delta_count
                    delta = batch_mean - old_mean
                    batch_total = batch_mean * delta_count
                    batch_second = (
                        current_var * total_count - old_var * old_count
                        - delta.square() * old_count * delta_count / total_count
                        + batch_mean.square() * delta_count
                    )
                self.all_reduce_(delta_count)
                self.all_reduce_(batch_total)
                self.all_reduce_(batch_second)
                count = old_count + delta_count
                total = old_mean * old_count + batch_total
                second = (old_var + old_mean.square()) * old_count + batch_second
                count_safe = count.clamp_min(1.0)
                mean = total / count_safe
                variance = (second / count_safe - mean.square()).clamp_min(1e-12)
            normalizer.running_mean.copy_(mean.to(normalizer.running_mean.dtype))
            normalizer.running_var.copy_(variance.to(normalizer.running_var.dtype))
            normalizer.count.copy_(count.to(normalizer.count.dtype))
            self._rms_snapshots[key] = (
                count.detach().clone(), mean.detach().clone(), variance.detach().clone(),
            )

    def gather(self, value: torch.Tensor) -> torch.Tensor:
        """Gather equal-shaped rank tensors, returning a concatenated tensor."""
        if not self.enabled:
            return value
        gathered = [torch.empty_like(value) for _ in range(self.world_size)]
        dist.all_gather(gathered, value)
        return torch.cat(gathered, dim=0)

    def metadata(self, *, reference_count_global: int, reference_count_local: int,
                 env_count_local: int) -> dict:
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "gradient_reduction": "all_reduce" if self.enabled else "none",
            "reference_sharding": "round_robin" if self.enabled else "none",
            "environment_sharding": "per_rank" if self.enabled else "single_process",
            "reference_count_global": int(reference_count_global),
            "reference_count_local": int(reference_count_local),
            "env_count_global": int(env_count_local * self.world_size),
            "env_count_local": int(env_count_local),
        }

    def close(self) -> None:
        if self.enabled and self.owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
