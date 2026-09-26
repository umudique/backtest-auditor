"""Phase 1 reporting contract tests."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from dataclasses import MISSING, fields
from typing import Any

import pandas as pd
import pytest

from src.contracts import AuditConfig, AuditReport, BacktestResult, MarketData, ValidationResult
from src.orchestrator import run_audit
from src.reporting.charts import PlotlyRenderer
from src.reporting.fragility import FragilityEvaluator
from src.reporting.report import AuditReportBuilder


def _backtest_result() -> BacktestResult:
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    gross_returns = pd.Series([0.02, 0.01, -0.01, 0.015], index=index)
    net_returns = pd.Series([0.015, 0.005, -0.015, 0.01], index=index)
    equity_curve = pd.Series([1.0, 1.015, 0.999775, 1.00977275], index=index)
    drawdown_series = pd.Series([0.0, 0.0, -0.015, -0.00515], index=index)
    return BacktestResult(
        positions=pd.Series([0.0, 1.0, 1.0, 0.0], index=index),
        trades=pd.DataFrame({"trade_size": [0.0, 1.0, 0.0, 1.0]}, index=index),
        gross_returns=gross_returns,
        net_returns=net_returns,
        equity_curve=equity_curve,
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
