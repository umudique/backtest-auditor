"""Phase 1 reporting contract tests."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.contracts import AuditConfig, AuditReport, BacktestResult, MarketData, ValidationResult
from src.orchestrator import run_audit, run_audit_from_file
from src.reporting.charts import PlotlyRenderer
from src.reporting.fragility import FragilityEvaluator
from src.reporting.report import AuditReportBuilder


def _backtest_result() -> BacktestResult:
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    gross_returns = pd.Series([0.02, 0.01, -0.01, 0.015], index=index)
    net_returns = pd.Series([0.015, 0.005, -0.015, 0.01], index=index)
    equity_curve = pd.Series([1.0, 1.015, 0.999775, 1.00977275], index=index)
    drawdown_series = pd.Series([0.0, 0.0, -0.015, -0.00515], index=index)
    net_equity_curve = (1 + net_returns).cumprod()
    return BacktestResult(
        positions=pd.Series([0.0, 1.0, 1.0, 0.0], index=index),
        trades=pd.DataFrame({"trade_size": [0.0, 1.0, 0.0, 1.0]}, index=index),
        gross_returns=gross_returns,
        net_returns=net_returns,
        equity_curve=equity_curve,
        net_equity_curve=net_equity_curve,
        drawdown_series=drawdown_series,
        execution_metadata={"timing": "signal_close_execute_next_open"},
    )


def _validation_result() -> ValidationResult:
    return ValidationResult(
        in_sample_metrics={"sharpe": 1.2},
        out_of_sample_metrics={"sharpe": 0.4},
        walk_forward_results=[
            {
                "train_start": 0,
                "train_end": 2,
                "evaluation_start": 2,
                "evaluation_end": 3,
                "result": 0.01,
            }
        ],
        sensitivity_results={"grid": [{"params": {"short_window": 2}, "result": 0.1}]},
        bootstrap_results={"random_seed": 7, "samples": [[0.01, -0.01]]},
        monte_carlo_results={"random_seed": 7, "paths": [[-0.01, 0.01]]},
    )


def _market_data() -> MarketData:
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    return MarketData(
        timestamp=pd.Series(index, index=index),
        open=pd.Series([10.0, 11.0, 12.0, 13.0], index=index),
        high=pd.Series([11.0, 12.0, 13.0, 14.0], index=index),
        low=pd.Series([9.0, 10.0, 11.0, 12.0], index=index),
        close=pd.Series([10.5, 11.5, 12.5, 13.5], index=index),
        volume=pd.Series([100, 110, 120, 130], index=index),
    )


def _audit_config() -> AuditConfig:
    return AuditConfig(
        strategy_name="moving_average_crossover",
        strategy_parameters={"short_window": 2, "long_window": 3},
        fee_assumption={"fee_rate": 0.001},
        slippage_assumption={"slippage_rate": 0.001},
        position_sizing={"min_position": -1.0, "max_position": 1.0},
        execution_assumptions={"timing": "signal_close_execute_next_open"},
        validation_configuration={"split_ratio": 0.6},
        random_seed=7,
    )


def test_audit_report_contains_all_required_fields() -> None:
    report_field_names = [field.name for field in fields(AuditReport)]

    assert report_field_names == [
        "baseline_metrics",
        "cost_adjusted_metrics",
        "out_of_sample_metrics",
        "walk_forward_summary",
        "parameter_sensitivity_summary",
        "simulated_drawdown_summary",
        "regime_summary",
        "fragility_summary",
        "verdict",
        "charts",
        "sensitivity_grid_rows",
    ]


def test_audit_report_verdict_has_no_default() -> None:
    verdict_field = next(field for field in fields(AuditReport) if field.name == "verdict")

    assert verdict_field.default is MISSING
    assert verdict_field.default_factory is MISSING
    with pytest.raises(TypeError):
        AuditReport()


def test_public_reporting_methods_have_docstring_contracts() -> None:
    package = importlib.import_module("src.reporting")
    modules = [
        importlib.import_module(module_info.name)
        for module_info in pkgutil.iter_modules(package.__path__, package.__name__ + ".")
    ]
    modules.append(importlib.import_module("src.orchestrator"))

    public_callables: list[Any] = []
    for module in modules:
        for _, member in inspect.getmembers(module):
            if inspect.isclass(member) and member.__module__ == module.__name__:
                public_callables.extend(
                    method
                    for name, method in inspect.getmembers(member, inspect.isfunction)
                    if not name.startswith("_")
                )
            elif inspect.isfunction(member) and member.__module__ == module.__name__:
                public_callables.append(member)

    assert public_callables
    for callable_object in public_callables:
        docstring = inspect.getdoc(callable_object)
        assert docstring is not None
        assert "Args:" in docstring
        assert "Returns:" in docstring
        assert "Raises:" in docstring
        assert "Invariants:" in docstring


def test_fragility_evaluator_emits_cost_deterioration_conclusion() -> None:
    conclusions = FragilityEvaluator().evaluate(
        _validation_result(),
        {"gross_sharpe": 1.4, "net_sharpe": 0.7},
    )

    assert any(
        "gross_sharpe" in conclusion and "net_sharpe" in conclusion for conclusion in conclusions
    )


def test_fragility_evaluator_emits_no_conclusion_without_supporting_metric() -> None:
    conclusions = FragilityEvaluator().evaluate(_validation_result(), {"net_sharpe": 0.7})

    assert conclusions == []


def test_audit_report_builder_is_deterministic_for_same_inputs() -> None:
    builder = AuditReportBuilder()
    backtest_result = _backtest_result()
    validation_result = _validation_result()
    metrics = {"gross_sharpe": 1.4, "net_sharpe": 0.7}
    fragility_summary = ["cost deterioration supported by gross_sharpe and net_sharpe"]
    charts = [{"kind": "equity"}, {"kind": "drawdown"}]

    first = builder.build(backtest_result, validation_result, metrics, fragility_summary, charts)
    second = builder.build(backtest_result, validation_result, metrics, fragility_summary, charts)

    assert isinstance(first, AuditReport)
    assert first.fragility_summary == second.fragility_summary
    assert first.verdict == second.verdict


def test_audit_report_builder_produces_robust_verdict_for_empty_fragility_summary() -> None:
    report = AuditReportBuilder().build(_backtest_result(), _validation_result(), {}, [], [])

    assert isinstance(report, AuditReport)
    assert report.verdict == "ROBUST"
    assert report.fragility_summary == []


def test_plotly_renderer_equity_curve_returns_chart_and_does_not_mutate() -> None:
    backtest_result = _backtest_result()
    before_equity = backtest_result.equity_curve.copy(deep=True)
    before_drawdown = backtest_result.drawdown_series.copy(deep=True)

    chart = PlotlyRenderer().render_equity_curve(backtest_result)

    assert chart is not None
    pd.testing.assert_series_equal(backtest_result.equity_curve, before_equity)
    pd.testing.assert_series_equal(backtest_result.drawdown_series, before_drawdown)


def test_orchestrator_run_audit_returns_audit_report_instance() -> None:
    report = run_audit(_market_data(), _audit_config())

    assert isinstance(report, AuditReport)


def test_orchestrator_does_not_compute_metrics_directly() -> None:
    import src.orchestrator as orchestrator_module

    source = inspect.getsource(orchestrator_module)

    assert "def sharpe_ratio" not in source
    assert "def sortino_ratio" not in source
    assert "def drawdown_series" not in source


def test_builder_separates_is_and_full_period_max_drawdown() -> None:
    """baseline_metrics must store IS max_drawdown under 'max_drawdown'
    and full-period under 'maximum_drawdown' so KPI and table use different values."""
    is_max_dd = -0.10
    full_period_max_dd = -0.22
    validation = ValidationResult(
        in_sample_metrics={"sharpe": 0.8, "net_return": 0.05, "max_drawdown": is_max_dd},
        out_of_sample_metrics={"sharpe": 0.4, "net_return": 0.02, "max_drawdown": -0.15},
        walk_forward_results=[],
        sensitivity_results={},
        bootstrap_results={"paths": []},
        monte_carlo_results={"paths": []},
    )
    metrics = {
        "gross_sharpe": 1.0,
        "net_sharpe": 0.7,
        "gross_cumulative": 0.18,
        "net_cumulative": 0.12,
        "net_return": 0.12,
        "max_drawdown": full_period_max_dd,
        "maximum_drawdown": full_period_max_dd,
        "average_drawdown": -0.08,
    }

    report = AuditReportBuilder().build(_backtest_result(), validation, metrics, [], [])

    assert report.baseline_metrics["max_drawdown"] == is_max_dd, (
        "baseline_metrics['max_drawdown'] must be the IS sub-period value for the table"
    )
    assert report.baseline_metrics["maximum_drawdown"] == full_period_max_dd, (
        "baseline_metrics['maximum_drawdown'] must be the full-period value for the KPI"
    )


def test_full_period_drawdown_is_at_least_as_severe_as_any_subperiod() -> None:
    """max drawdown over the full period must be <= max drawdown of IS or OOS sub-period."""
    from src.metrics.drawdown import DrawdownMetrics
    from src.validation.in_sample_oos import InSampleOutOfSampleValidator

    index = pd.date_range("2024-01-01", periods=20, freq="D")
    # equity that peaks mid-IS then drops through OOS: worst drawdown spans both periods
    equity = pd.Series(
        [
            1.0,
            1.05,
            1.12,
            1.08,
            1.10,
            1.15,
            1.10,
            1.05,
            1.00,
            0.97,
            0.95,
            0.92,
            0.90,
            0.88,
            0.87,
            0.89,
            0.91,
            0.90,
            0.92,
            0.93,
        ],
        index=index,
    )
    net_returns = equity.pct_change().fillna(0.0)
    net_equity = (1 + net_returns).cumprod()

    br = _backtest_result()
    from dataclasses import replace as dc_replace

    br = dc_replace(
        br,
        gross_returns=net_returns,
        net_returns=net_returns.copy(),
        equity_curve=equity,
        net_equity_curve=net_equity,
        drawdown_series=(equity / equity.cummax()) - 1,
        positions=pd.Series([1.0] * 20, index=index),
        trades=pd.DataFrame({"trade_size": [0.0] * 20}, index=index),
    )

    partitions = InSampleOutOfSampleValidator().split(br, split_ratio=0.6)
    dm = DrawdownMetrics()
    full_dd = dm.maximum_drawdown(br)
    is_dd = dm.maximum_drawdown(partitions["in_sample"])
    oos_dd = dm.maximum_drawdown(partitions["out_of_sample"])

    assert full_dd <= is_dd, f"full {full_dd:.4f} must be <= IS {is_dd:.4f}"
    assert full_dd <= oos_dd, f"full {full_dd:.4f} must be <= OOS {oos_dd:.4f}"


def test_metric_delta_equals_oos_minus_is_on_raw_values() -> None:
    """Delta must equal OOS − IS from raw values, not from rounded display values."""
    from app.display import _metric_delta

    report = AuditReport(
        baseline_metrics={"sharpe": -0.2801, "net_return": -0.0928, "max_drawdown": -0.1740},
        cost_adjusted_metrics={"net_return": -0.05, "net_sharpe": -0.30, "net_cumulative": -0.05},
        out_of_sample_metrics={"sharpe": -0.4408, "net_return": -0.0551, "max_drawdown": -0.1740},
        walk_forward_summary={},
        parameter_sensitivity_summary={},
        simulated_drawdown_summary={},
        regime_summary={},
        fragility_summary=[],
        verdict="ROBUST",
        charts=[],
        sensitivity_grid_rows=[],
    )

    for key in ("sharpe", "net_return", "max_drawdown"):
        expected = report.out_of_sample_metrics[key] - report.baseline_metrics[key]
        actual = _metric_delta(report, key)
        assert actual == pytest.approx(expected, abs=1e-9), (
            f"delta for '{key}': expected {expected:.6f}, got {actual:.6f}"
        )


def test_net_equity_curve_is_derived_from_net_returns_not_gross() -> None:
    """net_equity_curve must track net_returns, not the gross equity_curve."""
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    gross_returns = pd.Series([0.05, 0.03, -0.02, 0.04, 0.01], index=index)
    net_returns = pd.Series([0.04, 0.02, -0.03, 0.03, 0.00], index=index)
    gross_equity = (1 + gross_returns).cumprod()
    expected_net_equity = (1 + net_returns).cumprod()

    br = _backtest_result()
    from dataclasses import replace as dc_replace

    br = dc_replace(
        br,
        gross_returns=gross_returns,
        net_returns=net_returns.copy(),
        equity_curve=gross_equity,
        net_equity_curve=(1 + net_returns).cumprod(),
        drawdown_series=pd.Series([0.0] * 5, index=index),
        positions=pd.Series([1.0] * 5, index=index),
        trades=pd.DataFrame({"trade_size": [0.0] * 5}, index=index),
    )

    pd.testing.assert_series_equal(br.net_equity_curve, expected_net_equity)
    # must not equal gross equity
    assert not br.net_equity_curve.equals(br.equity_curve)


def test_fragility_narrow_plateau_threshold_is_exclusive() -> None:
    """Finding emitted when profitable ratio < 0.5, not when exactly == 0.5."""
    metrics = {"gross_sharpe": 1.0, "net_sharpe": 0.95}
    validation = ValidationResult(
        in_sample_metrics={"sharpe": 1.0},
        out_of_sample_metrics={"sharpe": 0.9},
        walk_forward_results=[],
        sensitivity_results={},
        bootstrap_results={},
        monte_carlo_results={},
    )

    # exactly 50% profitable — must NOT emit finding
    grid_half = [
        {"Short": 10, "Long": 50, "Sharpe": 0.5},
        {"Short": 10, "Long": 60, "Sharpe": -0.1},
    ]
    findings_half = FragilityEvaluator().evaluate(validation, metrics, grid_half)
    assert not any("narrow profitable plateau" in f for f in findings_half), (
        "50% profitable must not trigger narrow plateau finding (threshold is < 0.5)"
    )

    # 49% profitable (1 of 3 with rounding) — must emit finding
    grid_below = [
        {"Short": 10, "Long": 50, "Sharpe": 0.5},
        {"Short": 10, "Long": 60, "Sharpe": -0.1},
        {"Short": 10, "Long": 70, "Sharpe": -0.2},
    ]
    findings_below = FragilityEvaluator().evaluate(validation, metrics, grid_below)
    assert any("narrow profitable plateau" in f for f in findings_below), (
        "1/3 profitable (33%) must trigger narrow plateau finding"
    )


def test_plotly_renderer_drawdown_returns_chart_and_does_not_mutate() -> None:
    result = _backtest_result()
    original_dd = result.drawdown_series.copy()

    chart = PlotlyRenderer().render_drawdown(result)

    assert chart is not None
    assert result.drawdown_series.equals(original_dd)


def test_equity_curve_view_model_contains_equity_curve_key() -> None:
    from src.reporting.view_models import EquityCurveViewModel

    result = _backtest_result()
    view = EquityCurveViewModel().build(result)

    assert "equity_curve" in view
    assert len(view["equity_curve"]) == len(result.equity_curve)


def test_drawdown_view_model_contains_drawdown_key() -> None:
    from src.reporting.view_models import DrawdownViewModel

    result = _backtest_result()
    view = DrawdownViewModel().build(result)

    assert "drawdown" in view
    assert len(view["drawdown"]) == len(result.drawdown_series)


def test_sensitivity_heatmap_view_model_contains_sensitivity_key() -> None:
    from src.reporting.view_models import SensitivityHeatmapViewModel

    validation = _validation_result()
    view = SensitivityHeatmapViewModel().build(validation)

    assert "sensitivity" in view
    assert view["sensitivity"] == validation.sensitivity_results


def test_walk_forward_view_model_contains_results_key() -> None:
    from src.reporting.view_models import WalkForwardViewModel

    validation = _validation_result()
    view = WalkForwardViewModel().build(validation)

    assert "results" in view
    assert view["results"] == list(validation.walk_forward_results)


def test_monte_carlo_drawdown_view_model_contains_both_stochastic_keys() -> None:
    from src.reporting.view_models import MonteCarloDrawdownViewModel

    validation = _validation_result()
    view = MonteCarloDrawdownViewModel().build(validation)

    assert "bootstrap" in view
    assert "monte_carlo" in view
    assert view["bootstrap"] == validation.bootstrap_results
    assert view["monte_carlo"] == validation.monte_carlo_results


def test_run_audit_from_file_returns_audit_report_instance(tmp_path: Path) -> None:
    index = pd.date_range("2024-01-01", periods=60, freq="D")
    df = pd.DataFrame(
        {
            "timestamp": index,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1_000_000,
        },
    )
    csv_path = tmp_path / "prices.csv"
    df.to_csv(csv_path, index=False)

    report = run_audit_from_file(str(csv_path), _audit_config())

    assert isinstance(report, AuditReport)
