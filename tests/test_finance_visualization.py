import struct

import pytest


def ledger_rows(sequence_count=2, anchor_count=2):
    rows = []
    for sequence_id in range(sequence_count):
        for anchor_index in range(anchor_count):
            anchor_month = anchor_index + 1
            anchor_period = (
                f"{2020 + (anchor_month - 1) // 12:04d}-"
                f"{(anchor_month - 1) % 12 + 1:02d}"
            )
            for horizon in range(1, 11):
                predicted = 0.01 * horizon + 0.001 * sequence_id
                target = 0.012 * horizon
                target_month = anchor_month + horizon
                target_period = (
                    f"{2020 + (target_month - 1) // 12:04d}-"
                    f"{(target_month - 1) % 12 + 1:02d}"
                )
                rows.append({
                    "sequence_id": sequence_id, "symbol": f"S{sequence_id}",
                    "anchor_index": anchor_index, "anchor_period": anchor_period,
                    "target_period": target_period,
                    "horizon": horizon,
                    "predicted_normalized_return": predicted,
                    "target_normalized_return": target,
                    "predicted_log_return": predicted,
                    "target_log_return": target,
                    "predicted_cumulative_log_return": predicted * horizon,
                    "target_cumulative_log_return": target * horizon,
                    "reference_cumulative_log_return": target * horizon,
                    "latent_source": "oracle_future_encoder",
                    "encoder_input_mode": "clean",
                    "action_mode": "deterministic_mean",
                    "cache_reset": anchor_index == 0,
                    "cache_reset_reason": "sequence_start" if anchor_index == 0 else "",
                    "rollout_mode": "privileged_train_playback",
                })
    return rows


def test_render_trajectory_png_is_deterministic_and_has_expected_dimensions(tmp_path):
    from gear_sonic.finance.visualization import render_trajectory_png

    rows = ledger_rows()
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    first = render_trajectory_png(rows, first_path, title="test", max_sequences=2)
    second = render_trajectory_png(rows, second_path, title="test", max_sequences=2)
    assert first == second
    payload = first_path.read_bytes()
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", payload[16:24]) == (1600, 1000)
    assert payload == second_path.read_bytes()
    assert first["sequence_count"] == 2
    assert first["anchor_count"] == 4


@pytest.mark.parametrize(
    "mutation", ["missing", "nonfinite", "bad_horizon", "duplicate", "target_period"]
)
def test_render_trajectory_png_rejects_invalid_rows(tmp_path, mutation):
    from gear_sonic.finance.evaluation import TRAJECTORY_FIELDNAMES
    from gear_sonic.finance.visualization import render_trajectory_png

    rows = ledger_rows(sequence_count=1, anchor_count=1)
    if mutation == "missing":
        rows[0].pop("target_period")
    elif mutation == "nonfinite":
        rows[0]["predicted_log_return"] = float("nan")
    elif mutation == "bad_horizon":
        rows[0]["horizon"] = 11
    elif mutation == "duplicate":
        rows.append(dict(rows[0]))
    else:
        rows[1]["target_period"] = "2099-01"
    assert set(rows[0]) <= set(TRAJECTORY_FIELDNAMES)
    with pytest.raises(ValueError):
        render_trajectory_png(rows, tmp_path / "invalid.png")


def test_render_trajectory_png_limits_sequence_ids(tmp_path):
    from gear_sonic.finance.visualization import render_trajectory_png

    result = render_trajectory_png(
        ledger_rows(sequence_count=3, anchor_count=1), tmp_path / "limited.png", max_sequences=2,
    )
    assert result["sequence_count"] == 2
    assert result["anchor_count"] == 2


def test_render_trajectory_png_reports_non_string_extra_keys_as_value_error(tmp_path):
    from gear_sonic.finance.visualization import render_trajectory_png

    row = ledger_rows(sequence_count=1, anchor_count=1)[0]
    row[None] = "unexpected"
    with pytest.raises(ValueError, match="unknown fields"):
        render_trajectory_png([row], tmp_path / "invalid-extra.png")


def test_render_trajectory_png_reports_fixed_canvas_downsampling(tmp_path):
    from gear_sonic.finance.visualization import render_trajectory_png

    result = render_trajectory_png(
        ledger_rows(sequence_count=5, anchor_count=14),
        tmp_path / "downsampled.png",
        max_sequences=5,
    )
    assert result["sequence_count"] == 5
    assert result["visualized_sequence_count"] == 4
    assert result["path_sequence_downsampled"] is True
    assert result["anchor_count"] == 70
    assert result["visualized_anchor_count"] == 48
    assert result["path_anchor_downsampled"] is True
    assert (result["heatmap_rows_rendered"], result["heatmap_rows_total"]) == (25, 56)
    assert result["heatmap_downsampled"] is True
    assert (result["direction_rows_rendered"], result["direction_rows_total"]) == (36, 56)
    assert result["direction_downsampled"] is True


def test_render_trajectory_png_preserves_absolute_magnitude_in_labels(tmp_path):
    from gear_sonic.finance.visualization import render_trajectory_png

    rows = ledger_rows(sequence_count=1, anchor_count=1)
    scaled = [dict(row) for row in rows]
    for row in scaled:
        for field in (
            "predicted_normalized_return", "target_normalized_return",
            "predicted_log_return", "target_log_return",
            "predicted_cumulative_log_return", "target_cumulative_log_return",
            "reference_cumulative_log_return",
        ):
            row[field] *= 100.0
    first = tmp_path / "magnitude-1.png"
    second = tmp_path / "magnitude-100.png"
    render_trajectory_png(rows, first)
    render_trajectory_png(scaled, second)
    assert first.read_bytes() != second.read_bytes()
