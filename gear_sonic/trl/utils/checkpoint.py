"""Checkpoint migration helpers with explicit compatibility allowlists."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


def load_state_dict_with_allowed_missing_prefixes(
    module: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    allowed_missing_prefixes: Sequence[str],
):
    """Allow only explicitly named new parameters during weight transfer."""
    prefixes = tuple(str(prefix) for prefix in allowed_missing_prefixes)
    if not prefixes or any(not prefix for prefix in prefixes):
        raise ValueError("allowed_missing_prefixes must contain non-empty prefixes")

    # strict=False still raises on same-name tensors with different shapes.
    result = module.load_state_dict(state_dict, strict=False)
    disallowed_missing = sorted(
        key for key in result.missing_keys if not key.startswith(prefixes)
    )
    unexpected = sorted(result.unexpected_keys)
    if disallowed_missing or unexpected:
        details = []
        if disallowed_missing:
            details.append(f"disallowed missing keys: {disallowed_missing}")
        if unexpected:
            details.append(f"unexpected checkpoint keys: {unexpected}")
        raise RuntimeError(
            "Checkpoint is not compatible with the declared migration: "
            + "; ".join(details)
        )
    return result
