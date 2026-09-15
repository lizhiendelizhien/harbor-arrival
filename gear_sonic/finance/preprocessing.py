"""Fixed, auditable median/IQR transforms for reference-trajectory features."""

from __future__ import annotations

import numpy as np


def fit_feature_statistics(values, fields, clip: float = 10.0) -> dict:
    """Fit each feature independently; absent cells are not numerical zeros.

    The supplied reference pool defines the fit scope. Each column is processed
    separately to support large memory-mapped future arrays without a full copy.
    """
    values = np.asarray(values)
    fields = list(fields)
    if values.ndim < 2 or values.shape[-1] != len(fields) or not fields:
        raise ValueError("Unexpected feature schema")
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError("clip must be finite and positive")
    result = {"fields": fields, "clip": float(clip), "center": [], "scale": [],
              "features": {}}
    for index, field in enumerate(fields):
        column = values[..., index].reshape(-1)
        valid = column[np.isfinite(column)]
        if not len(valid):
            raise ValueError(f"Every feature needs valid reference values: {field}")
        q = np.quantile(valid, [0, .005, .25, .5, .75, .995, 1])
        center = float(q[3])
        spread = float(q[4] - q[2])
        scale = spread if spread > 1e-6 else 1.0
        lower, upper = center - clip * scale, center + clip * scale
        below, above = int((valid < lower).sum()), int((valid > upper).sum())
        result["center"].append(center)
        result["scale"].append(scale)
        result["features"][field] = {
            "valid_count": len(valid), "missing_or_nonfinite_count": len(column) - len(valid),
            "min": float(q[0]), "p005": float(q[1]), "q25": float(q[2]),
            "median": center, "q75": float(q[4]), "p995": float(q[5]), "max": float(q[6]),
            "iqr": spread, "scale": scale, "constant_scale_fallback": spread <= 1e-6,
            "lower_bound": lower, "upper_bound": upper,
            "below_bound": below, "above_bound": above,
            "clipped_fraction": (below + above) / len(valid),
        }
    return result


def transform_features(values, stats: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return bounded model inputs and clipping flags, without changing targets."""
    values = np.asarray(values)
    center = np.asarray(stats["center"], dtype=np.float64)
    scale = np.asarray(stats["scale"], dtype=np.float64)
    clip = float(stats["clip"])
    if values.shape[-1] != len(stats["fields"]) or center.shape != scale.shape or center.shape != (values.shape[-1],):
        raise ValueError("Unexpected feature schema")
    if not np.isfinite(values).all():
        raise ValueError("Feature values must be finite; handle missing masks explicitly")
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or (scale <= 0).any() or not np.isfinite(clip) or clip <= 0:
        raise ValueError("Invalid normalization statistics")
    normalized = (values.astype(np.float64) - center) / scale
    clipped = (normalized < -clip) | (normalized > clip)
    return normalized.clip(-clip, clip).astype(np.float32), clipped
