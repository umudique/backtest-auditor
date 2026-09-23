"""Phase 1 tests for metrics-layer contracts."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from src.contracts import BacktestResult
from src.metrics.aggregator import MetricsAggregator
from src.metrics.drawdown import DrawdownMetrics
from src.metrics.regime import RegimeAnalyzer
from src.metrics.returns import ReturnMetrics


def backtest_result_fixture(
    net_returns: list[float] | pd.Series,
    equity_curve: list[float] | pd.Series | None = None,
) -> BacktestResult:
    returns = pd.Series(net_returns, dtype=float)
    index = pd.RangeIndex(len(returns))
    returns.index = index
    gross_returns = returns.copy()
    if equity_curve is None:
        equity = (1.0 + returns).cumprod()
    else:
        equity = pd.Series(equity_curve, dtype=float, index=index)

    return BacktestResult(
        positions=pd.Series([0.0] * len(returns), index=index),
        trades=pd.DataFrame({"trade_size": [0.0] * len(returns)}, index=index),
        gross_returns=gross_returns,
        net_returns=returns.copy(),
        equity_curve=equity,
        drawdown_series=pd.Series([0.0] * len(returns), index=index),
        execution_metadata={"timing": "signal_close_execute_next_open"},
    )


@settings(max_examples=50)
@given(values=st.lists(st.floats(min_value=-0.2, max_value=0.2), min_size=2, max_size=30))
def test_sharpe_ratio_is_zero_when_mean_excess_return_is_zero(values: list[float]) -> None:
    returns = pd.Series(values, dtype=float)
    demeaned = returns - returns.mean()
    assume(float(demeaned.std(ddof=1)) > 0)
    result = backtest_result_fixture(demeaned)

    sharpe = ReturnMetrics().sharpe_ratio(
        result,
        risk_free_rate=0.0,
        annualization_factor=252.0,
    )

    assert sharpe == pytest.approx(0.0)


@settings(max_examples=50)
@given(values=st.lists(st.floats(min_value=-0.2, max_value=0.3), min_size=3, max_size=40))
def test_sortino_ratio_is_at_least_sharpe_when_downside_deviation_is_lower(
    values: list[float],
) -> None:
    returns = pd.Series(values, dtype=float)
    assume(float(returns.mean()) > 0)
    assume((returns < 0).any())
    downside = returns[returns < 0]
    assume(float(downside.std(ddof=1)) > 0)
    assume(float(downside.std(ddof=1)) <= float(returns.std(ddof=1)))
    result = backtest_result_fixture(returns)

    sharpe = ReturnMetrics().sharpe_ratio(result, risk_free_rate=0.0, annualization_factor=1.0)
    sortino = ReturnMetrics().sortino_ratio(result, target_return=0.0, annualization_factor=1.0)

    assert sortino >= sharpe


@settings(max_examples=50)
@given(values=st.lists(st.floats(min_value=0.01, max_value=10.0), min_size=1, max_size=40))
def test_maximum_drawdown_is_never_positive(values: list[float]) -> None:
    result = backtest_result_fixture([0.0] * len(values), equity_curve=values)

    max_drawdown = DrawdownMetrics().maximum_drawdown(result)

    assert max_drawdown <= 0


@settings(max_examples=50)
@given(values=st.lists(st.floats(min_value=0.01, max_value=10.0), min_size=1, max_size=40))
def test_drawdown_series_never_exceeds_zero(values: list[float]) -> None:
    result = backtest_result_fixture([0.0] * len(values), equity_curve=values)

    drawdowns = DrawdownMetrics().drawdown_series(result)

    assert (drawdowns <= 0).all()


def test_sharpe_ratio_known_inputs() -> None:
    result = backtest_result_fixture([0.01, 0.02, 0.03])
    excess = result.net_returns
    expected = excess.mean() / excess.std(ddof=1) * math.sqrt(252.0)

    actual = ReturnMetrics().sharpe_ratio(
        result,
        risk_free_rate=0.0,
        annualization_factor=252.0,
    )

    assert actual == pytest.approx(expected)


def test_sortino_ratio_known_inputs() -> None:
    result = backtest_result_fixture([0.02, -0.01, 0.03, -0.02])
    downside = result.net_returns[result.net_returns < 0]
    expected = result.net_returns.mean() / downside.std(ddof=1) * math.sqrt(252.0)

    actual = ReturnMetrics().sortino_ratio(
        result,
        target_return=0.0,
        annualization_factor=252.0,
    )

    assert actual == pytest.approx(expected)


def test_expectancy_known_inputs() -> None:
    result = backtest_result_fixture([0.02, 0.04, -0.01, -0.03])
    wins = pd.Series([0.02, 0.04])
    losses = pd.Series([-0.01, -0.03])
    expected = (2 / 4) * wins.mean() - (2 / 4) * abs(losses.mean())

    actual = ReturnMetrics().expectancy(result)

    assert actual == pytest.approx(expected)


def test_maximum_drawdown_known_equity_curve() -> None:
    result = backtest_result_fixture([0.0, 0.0, 0.0, 0.0], equity_curve=[1.0, 1.1, 0.9, 1.05])
    expected = (0.9 / 1.1) - 1.0

    actual = DrawdownMetrics().maximum_drawdown(result)

    assert actual == pytest.approx(expected)


def test_sharpe_zero_denominator_never_defaults_to_one() -> None:
    result = backtest_result_fixture([0.01, 0.01, 0.01])

    with pytest.raises(ValueError, match="standard deviation|denominator|zero"):
        ReturnMetrics().sharpe_ratio(result, risk_free_rate=0.0, annualization_factor=252.0)


def test_sortino_zero_downside_denominator_never_defaults_to_sharpe() -> None:
    result = backtest_result_fixture([0.01, 0.02, 0.03])

    with pytest.raises(ValueError, match="downside|denominator|zero"):
        ReturnMetrics().sortino_ratio(result, target_return=0.0, annualization_factor=252.0)


def test_regime_analyzer_rejects_missing_external_labels() -> None:
    result = backtest_result_fixture([0.01, -0.02, 0.03])

    with pytest.raises(ValueError, match="regime|labels|external"):
        RegimeAnalyzer().compare_by_regime(result, regime_labels=None)


def test_metrics_aggregator_outputs_flat_dict_of_floats() -> None:
    metrics = MetricsAggregator().aggregate(
        return_metrics={"sharpe": 1.2, "sortino": 1.8, "expectancy": 0.03},
        drawdown_metrics={"max_drawdown": -0.2, "average_drawdown": -0.05},
        regime_metrics={"regime_bull": 0.04, "regime_bear": -0.01},
    )

    assert isinstance(metrics, dict)
    assert all(isinstance(key, str) for key in metrics)
    assert all(isinstance(value, float) for value in metrics.values())
    assert all(not isinstance(value, dict | list | tuple) for value in metrics.values())
