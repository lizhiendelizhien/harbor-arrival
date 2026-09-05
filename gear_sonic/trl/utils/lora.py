"""Small, dependency-free LoRA implementation for Stage5 fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf
import torch
from torch import nn


class LoRALinear(nn.Module):
    """Frozen Linear projection plus a trainable low-rank residual."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 32,
        alpha: float | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if isinstance(base_layer.weight, torch.nn.parameter.UninitializedParameter):
            raise ValueError("LoRA target contains an unmaterialized LazyLinear layer")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = float(alpha if alpha is not None else rank)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base_layer.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.lora_B(self.lora_A(self.dropout(values))) * self.scaling
        return self.base_layer(values) + residual


@dataclass
class LoRAStats:
    replaced_linear_layers: int
    trainable_params: int
    total_params: int


def _to_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, ListConfig):
        return list(value)
    if isinstance(value, (tuple, list)):
        return list(value)
    return [value]


def _cfg_to_container(cfg):
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


def _get_submodule(root: nn.Module, module_path: str) -> nn.Module:
    module = root
    if module_path in ("", "."):
        return module
    for part in module_path.split("."):
        if not hasattr(module, part):
            raise ValueError(
                f"LoRA target '{module_path}' does not exist at component '{part}'"
            )
        module = getattr(module, part)
    return module


def _replace_linear_layers(
    module: nn.Module, rank: int, alpha: float, dropout: float
) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear):
            replacement = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
            replacement.to(device=child.weight.device, dtype=child.weight.dtype)
            setattr(module, name, replacement)
            replaced += 1
        else:
            replaced += _replace_linear_layers(child, rank, alpha, dropout)
    return replaced


def freeze_all_parameters(*modules: nn.Module | None) -> None:
    for module in modules:
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = False


def count_parameters(*modules: nn.Module | None) -> tuple[int, int]:
    trainable = 0
    total = 0
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            total += parameter.numel()
            if parameter.requires_grad:
                trainable += parameter.numel()
    return trainable, total


def _unfreeze_matching(module: nn.Module | None, patterns: list[str]) -> None:
    if module is None:
        return
    for name, parameter in module.named_parameters():
        if any(pattern in name for pattern in patterns):
            parameter.requires_grad = True


def apply_lora_adaptation(
    policy: nn.Module, value_model: nn.Module | None, cfg
) -> LoRAStats:
    """Freeze both models and inject LoRA into the configured Linear subtrees."""
    cfg = _cfg_to_container(cfg)
    if not cfg or not cfg.get("enabled", False):
        trainable, total = count_parameters(policy, value_model)
        return LoRAStats(0, trainable, total)

    rank = int(cfg.get("rank", 32))
    alpha = float(cfg.get("alpha", rank))
    dropout = float(cfg.get("dropout", 0.0))
    policy_targets = _to_list(cfg.get("policy_target_modules", []))
    value_targets = _to_list(cfg.get("value_target_modules", []))
    if not policy_targets:
        raise ValueError("LoRA is enabled but policy_target_modules is empty")

    freeze_all_parameters(policy, value_model)
    replaced = 0
    for module_path in policy_targets:
        count = _replace_linear_layers(
            _get_submodule(policy, module_path), rank, alpha, dropout
        )
        if count == 0:
            raise ValueError(f"LoRA policy target '{module_path}' contains no Linear layers")
        replaced += count
        logger.info(f"Injected LoRA into policy.{module_path}: {count} Linear layers")

    if value_model is not None:
        for module_path in value_targets:
            count = _replace_linear_layers(
                _get_submodule(value_model, module_path), rank, alpha, dropout
            )
            if count == 0:
                raise ValueError(f"LoRA value target '{module_path}' contains no Linear layers")
            replaced += count
            logger.info(f"Injected LoRA into value_model.{module_path}: {count} Linear layers")

    if bool(cfg.get("train_noise_std", True)):
        for name in ("std", "log_std"):
            parameter = getattr(policy, name, None)
            if isinstance(parameter, nn.Parameter):
                parameter.requires_grad = True

    patterns = _to_list(cfg.get("extra_trainable_patterns", []))
    _unfreeze_matching(policy, patterns)
    _unfreeze_matching(value_model, patterns)

    trainable, total = count_parameters(policy, value_model)
    if trainable == 0:
        raise RuntimeError("LoRA injection produced no trainable parameters")
    logger.info(
        "LoRA adaptation ready: "
        f"rank={rank}, alpha={alpha}, dropout={dropout}, layers={replaced}, "
        f"trainable={trainable}/{total} ({trainable / max(total, 1):.4%})"
    )
    return LoRAStats(replaced, trainable, total)


__all__ = [
    "LoRALinear",
    "LoRAStats",
    "apply_lora_adaptation",
    "count_parameters",
    "freeze_all_parameters",
]
