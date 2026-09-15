import csv
import math
from dataclasses import replace

import pytest

from gear_sonic.finance.observations import (
    CURRENT_FIELDS,
    FUTURE_FIELDS,
    MonthlyBar,
    build_market_context,
    build_symbol_observations,
    iter_sequences,
    load_observations,
    make_sample,
)


def bars(symbol="AAA", count=60, rate=0.02, volume=1000.0):
    result = []
    for index in range(count):
        price = 100 * (1 + rate) ** index
        result.append(MonthlyBar(
            symbol=symbol, period=f"{2020 + index // 12:04d}-{index % 12 + 1:02d}",
            open=price / 1.01, close=price, high=price * 1.03, low=price * 0.97,
            raw_close=price, volume=volume,
        ))
    return result


def observations(count=60):
    first = bars(count=count)
    market = build_market_context([*first, *bars("BBB", count=count, rate=-0.01)])
    return build_symbol_observations(first, market)


def test_feature_contract_and_future_relative_values():
    rows = observations()
    sample = make_sample(rows, 3)
    assert len(CURRENT_FIELDS) == 16
    assert len(FUTURE_FIELDS) == 15
    assert len(sample["current_state"]) == 16
    assert len(sample["future_reference"]) == 10
    for k, frame in enumerate(sample["future_reference"], start=1):
        assert len(frame) == 15
        assert frame[0] == pytest.approx(math.log1p(0.02))
        assert frame[1] == pytest.approx(k * math.log1p(0.02))
        assert frame[6] == pytest.approx(0, abs=1e-12)
        assert frame[10] == pytest.approx(0)
    state = sample["current_state"]
    assert state[5] == pytest.approx(1.02**3 - 1)
    assert state[6] == pytest.approx(1.02**6 - 1)
    assert state[14] == pytest.approx(0.005)
    assert state[15] == pytest.approx(1.02**6 - 1.005**6)


def test_future_prices_cannot_change_current_state_or_past_market_context():
    first, second = bars(), bars("BBB", rate=-0.01)
    original = build_symbol_observations(first, build_market_context([*first, *second]))
    changed = [replace(row, close=row.close * 5, high=row.high * 5, raw_close=row.raw_close * 5)
               if row.period > "2022-01" else row for row in first]
    updated = build_symbol_observations(changed, build_market_context([*changed, *second]))
    before = {row.period: row.current_state for row in original if row.period <= "2022-01"}
    after = {row.period: row.current_state for row in updated if row.period <= "2022-01"}
    assert before == after


def test_liquidity_ties_use_average_rank():
    first, second = bars(), bars("BBB")
    market = build_market_context([*first, *second])
    assert market["2021-01"].liquidity_ranks == {"AAA": 0.5, "BBB": 0.5}


def test_gaps_cannot_be_crossed_by_future_windows_or_sequences():
    rows = observations()
    missing = rows[:15] + rows[16:]
    with pytest.raises(ValueError, match="contiguous"):
        make_sample(missing, 10)
    batches = list(iter_sequences({"AAA": missing}, sequence_length=4))
    assert batches
    for batch in batches:
        months = [int(p[:4]) * 12 + int(p[5:]) for p in batch["periods"]]
        assert all(right - left == 1 for left, right in zip(months, months[1:]))


def test_splits_require_entire_future_window_inside_end_period():
    batches = list(iter_sequences(
        {"AAA": observations()}, sequence_length=4,
        start_period="2021-01", end_period="2023-12",
    ))
    assert batches
    for batch in batches:
        assert batch["periods"][0] >= "2021-01"
        assert batch["target_end_period"] <= "2023-12"
        assert len(batch["current_state"]) == 4


def test_duplicate_months_are_rejected():
    first = bars()
    with pytest.raises(ValueError, match="duplicate"):
        build_market_context([*first, first[0]])


@pytest.fixture
def canonical_file(tmp_path):
    path = tmp_path / "monthly_canonical.csv"
    fields = ["symbol", "period", "raw_valid", "qfq_valid", "qfq_open", "qfq_close",
              "qfq_high", "qfq_low", "raw_close", "raw_volume", "raw_amount"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for bar in [*bars(), *bars("BBB", rate=-0.01)]:
            writer.writerow({
                "symbol": bar.symbol, "period": bar.period, "raw_valid": 1, "qfq_valid": 1,
                "qfq_open": bar.open, "qfq_close": bar.close, "qfq_high": bar.high,
                "qfq_low": bar.low, "raw_close": bar.raw_close * 2,
                "raw_volume": bar.volume, "raw_amount": 0,
            })
    return path


def test_loading_one_symbol_still_uses_whole_market_and_zero_amount_is_allowed(canonical_file):
    loaded = load_observations(canonical_file, {"AAA"})
    assert set(loaded) == {"AAA"}
    assert loaded["AAA"][0].current_state[14] == pytest.approx(0.005)
    assert loaded["AAA"][0].current_state[0] == pytest.approx(0.02)


def test_real_file_pipeline_check(canonical_file):
    from scripts.check_finance_sonic import run_check

    report = run_check(
        canonical_file, symbols={"AAA"}, sequence_length=4, fit_start="2020-01",
        fit_end="2022-06", check_start="2022-07", check_end="2024-12", tiny=True,
    )
    assert report["input_shapes"]["current"] == [1, 4, 16]
    assert report["input_shapes"]["future"] == [1, 4, 10, 15]
    assert report["encoder_gradient_norm"] > 0
    assert report["streaming_max_abs_error"] < 1e-5


def test_pipeline_check_defaults_to_full_reference_pool(canonical_file):
    from scripts.check_finance_sonic import run_check

    report = run_check(canonical_file, symbols={"AAA", "BBB"}, sequence_length=4, tiny=True)
    assert report["dataset_partition"] == "none"
    assert report["fit_period"] == [None, None]
    assert report["check_period"] == [None, None]
    expected = list(iter_sequences(load_observations(canonical_file), sequence_length=4))
    assert report["fit_sequence_count"] == len(expected)
    assert report["input_shapes"]["current"] == [2, 4, 16]
    assert report["input_shapes"]["future"] == [2, 4, 10, 15]
    assert all(math.isfinite(loss) for loss in report["fit_batch_losses"].values())
    assert math.isfinite(report["encoder_gradient_norm"])
    assert report["encoder_gradient_norm"] > 0
    assert report["streaming_max_abs_error"] < 1e-5
