"""Phase 1 tests for validation-layer contracts."""

from __future__ import annotations

from typing import Any

import pandas as pd
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from src.contracts import BacktestResult, ValidationResult
from src.validation.aggregator import ValidationResultAggregator
from src.validation.bootstrap import BootstrapAnalyzer
from src.validation.in_sample_oos import InSampleOutOfSampleValidator
from src.validation.monte_carlo import MonteCarloReshuffler
from src.validation.sensitivity import ParameterSensitivityAnalyzer
from src.validation.walk_forward import WalkForwardValidator


def backtest_result_fixture(length: int = 10) -> BacktestResult:
    index = pd.date_range("2024-01-02", periods=length, freq="D")
    gross_returns = pd.Series([0.01 * ((i % 3) - 1) for i in range(length)], index=index)
    net_returns = gross_returns - 0.001
    equity_curve = (1.0 + net_returns).cumprod()
    drawdown_series = (equity_curve / equity_curve.cummax()) - 1.0
    positions = pd.Series([float(i % 2) for i in range(length)], index=index)
    trades = pd.DataFrame({"trade_size": positions.diff().fillna(positions).abs()}, index=index)

    net_equity_curve = (1.0 + net_returns).cumprod()
    return BacktestResult(
        positions=positions,
        trades=trades,
        gross_returns=gross_returns,
        net_returns=net_returns.copy(),
        equity_curve=equity_curve,
        net_equity_curve=net_equity_curve,
        drawdown_series=drawdown_series,
        execution_metadata={"timing": "signal_close_execute_next_open"},
    )


def assert_partitions_cover_without_gap_or_overlap(
    original: BacktestResult,
    in_sample: BacktestResult,
    out_of_sample: BacktestResult,
) -> None:
    original_index = list(original.net_returns.index)
    combined_index = list(in_sample.net_returns.index) + list(out_of_sample.net_returns.index)

    assert combined_index == original_index
    assert set(in_sample.net_returns.index).isdisjoint(set(out_of_sample.net_returns.index))
    assert in_sample.net_returns.index[-1] < out_of_sample.net_returns.index[0]


@settings(max_examples=50)
@given(
    length=st.integers(min_value=5, max_value=50),
    split_ratio=st.floats(min_value=0.2, max_value=0.8, allow_nan=False, allow_infinity=False),
)
def test_in_sample_oos_split_covers_period_without_gap_or_overlap(
    length: int,
    split_ratio: float,
) -> None:
    result = backtest_result_fixture(length)

    partitions = InSampleOutOfSampleValidator().split(result, split_ratio)

    assert set(partitions) == {"in_sample", "out_of_sample"}
    assert_partitions_cover_without_gap_or_overlap(
        result,
        partitions["in_sample"],
        partitions["out_of_sample"],
    )


@settings(max_examples=50)
@given(
    length=st.integers(min_value=12, max_value=80),
    train_size=st.integers(min_value=3, max_value=12),
    evaluation_size=st.integers(min_value=1, max_value=8),
    step_size=st.integers(min_value=1, max_value=12),
)
def test_walk_forward_evaluation_windows_do_not_overlap(
    length: int,
    train_size: int,
    evaluation_size: int,
    step_size: int,
) -> None:
    assume(train_size + evaluation_size <= length)
    result = backtest_result_fixture(length)

    windows = WalkForwardValidator().construct_windows(
        result,
        train_size=train_size,
        evaluation_size=evaluation_size,
        step_size=step_size,
    )

    evaluation_ranges = [
        range(window["evaluation_start"], window["evaluation_end"]) for window in windows
    ]
    for left_index, left in enumerate(evaluation_ranges):
        for right in evaluation_ranges[left_index + 1 :]:
            assert set(left).isdisjoint(set(right))


def test_bootstrap_fixed_seed_is_reproducible() -> None:
    result = backtest_result_fixture()

    first = BootstrapAnalyzer().run(result, random_seed=123, n_samples=25)
    second = BootstrapAnalyzer().run(result, random_seed=123, n_samples=25)

    assert first == second
    assert first["random_seed"] == 123


def test_monte_carlo_fixed_seed_is_reproducible() -> None:
    result = backtest_result_fixture()

    first = MonteCarloReshuffler().reshuffle(result, random_seed=456, n_paths=25)
    second = MonteCarloReshuffler().reshuffle(result, random_seed=456, n_paths=25)

    assert first == second
    assert first["random_seed"] == 456


def test_parameter_sensitivity_returns_diagnostic_grid_without_recommendations() -> None:
    result = backtest_result_fixture()

    sensitivity = ParameterSensitivityAnalyzer().analyze(
        result,
        parameter_grid={"short_window": [2, 3], "long_window": [5, 8]},
    )

    assert "grid" in sensitivity
    forbidden_keys = {"best", "recommended", "optimal"}
    assert forbidden_keys.isdisjoint(sensitivity)


def test_validation_result_aggregator_keeps_is_and_oos_metrics_separate() -> None:
    validation_result = ValidationResultAggregator().aggregate(
        in_sample_metrics={"sharpe": 1.2},
        out_of_sample_metrics={"sharpe": 0.4},
        walk_forward_results=[{"evaluation_start": 6, "evaluation_end": 8}],
        sensitivity_results={"grid": [{"short_window": 2, "long_window": 5}]},
        bootstrap_results={"random_seed": 123, "samples": [0.1, -0.1]},
        monte_carlo_results={"random_seed": 456, "paths": [0.05, -0.05]},
    )

    assert isinstance(validation_result, ValidationResult)
    assert validation_result.in_sample_metrics == {"sharpe": 1.2}
    assert validation_result.out_of_sample_metrics == {"sharpe": 0.4}
    assert validation_result.in_sample_metrics is not validation_result.out_of_sample_metrics
    assert "metrics" not in validation_result.__dict__
    assert "combined_metrics" not in validation_result.__dict__
    assert "merged_metrics" not in validation_result.__dict__


def test_known_sixty_forty_partition_exact_index_positions() -> None:
    result = backtest_result_fixture(length=10)

    partitions = InSampleOutOfSampleValidator().split(result, split_ratio=0.6)

    assert list(partitions["in_sample"].net_returns.index) == list(result.net_returns.index[:6])
    assert list(partitions["out_of_sample"].net_returns.index) == list(result.net_returns.index[6:])


def test_validation_components_consume_backtest_result_only() -> None:
    result = backtest_result_fixture()

    assert isinstance(result, BacktestResult)


def assert_no_recommendation_keys(mapping: dict[str, Any]) -> None:
    assert {"best", "recommended", "optimal"}.isdisjoint(mapping)
