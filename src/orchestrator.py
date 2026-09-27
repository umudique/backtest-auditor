"""Audit workflow orchestration contract."""

from dataclasses import replace
from typing import Any, cast

import pandas as pd

from src.contracts import AuditConfig, AuditReport, BacktestResult, MarketData
from src.data.canonical import Canonicalizer
from src.data.loader import MarketDataLoader
from src.data.schema import SchemaValidator
from src.data.validation import TimeSeriesValidator, ValueValidator
from src.engine.costs import CostModel
from src.engine.execution import ExecutionModel
from src.engine.position_sizing import PositionSizer
from src.engine.result import BacktestResultBuilder
from src.engine.simulator import PortfolioSimulator
from src.engine.strategy import MovingAverageCrossoverStrategy
from src.metrics.aggregator import MetricsAggregator
from src.metrics.drawdown import DrawdownMetrics
from src.metrics.returns import ReturnMetrics
from src.reporting.charts import PlotlyRenderer
from src.reporting.fragility import FragilityEvaluator
from src.reporting.report import AuditReportBuilder
from src.validation.aggregator import ValidationResultAggregator
from src.validation.bootstrap import BootstrapAnalyzer
from src.validation.in_sample_oos import InSampleOutOfSampleValidator
from src.validation.monte_carlo import MonteCarloReshuffler
from src.validation.sensitivity import ParameterSensitivityAnalyzer
from src.validation.walk_forward import WalkForwardValidator


def run_audit_from_file(file_path: str, config: AuditConfig) -> AuditReport:
    """Load market data from a validated path and run the audit pipeline.

    Args:
        file_path: CSV or Parquet path already accepted by the UI surface-level
            upload validator.
        config: Full audit configuration collected by the UI.

    Returns:
        A canonical ``AuditReport`` produced by ``run_audit``.

    Raises:
        ValueError: If data loading, validation, canonicalization, or audit
            execution rejects the supplied file or configuration.
        FileNotFoundError: If the file path does not exist.
        OSError: If the supported file cannot be read.

    Invariants:
        File loading and canonicalization are owned below the UI boundary.
        This helper delegates analytical execution to ``run_audit`` and does
        not duplicate orchestration logic.
    """
    data = MarketDataLoader().load(file_path)
    data = SchemaValidator().validate(data)
    data = TimeSeriesValidator().validate(data)
    data = ValueValidator().validate(data)
    market_data = Canonicalizer().canonicalize(data)
    return run_audit(market_data, config)


def run_audit(market_data: MarketData, config: AuditConfig) -> AuditReport:
    """Coordinate the complete backtest audit pipeline.

    Args:
        market_data: Canonical market data produced by the data layer.
        config: Full audit configuration containing strategy, cost, execution,
            validation, and stochastic reproducibility assumptions.

    Returns:
        A canonical ``AuditReport`` assembled by the reporting layer.

    Raises:
        NotImplementedError: Until Phase 2 wires the full pipeline.
        ValueError: If pipeline inputs are malformed, if a required stage
            cannot produce its canonical output, or if a report verdict would
            lack explicit supporting evidence.

    Invariants:
        Orchestration contains no financial logic, calculations, metric
        formulas, strategy logic, fee application, validation algorithms, or UI
        behavior.
        The orchestrator may coordinate data, engine, validation, metrics, and
        reporting components, but never imports from ``app``.
        Domain contracts are imported only from ``src.contracts`` and are not
        redefined.
    """
    signals = MovingAverageCrossoverStrategy().generate_signals(
        market_data,
        config.strategy_parameters,
    )
    sized_positions = PositionSizer().size_positions(
        signals, cast("dict[str, Any]", config.position_sizing)
    )
    executed_positions, execution_metadata = ExecutionModel().apply_execution_timing(
        sized_positions,
        market_data.open,
        config.execution_assumptions,
    )

    simulation_result = PortfolioSimulator().simulate(
        market_data,
        executed_positions,
        {"initial_equity": 1.0},
    )
    positions = cast(pd.Series, simulation_result["positions"])
    trades = cast(pd.DataFrame, simulation_result["trades"])
    gross_returns = cast(pd.Series, simulation_result["gross_returns"])
    cost_config = _cost_config(config)
    net_returns = CostModel().apply_costs(gross_returns, trades, cost_config)

    backtest_result = BacktestResultBuilder().build(
        positions,
        trades,
        gross_returns,
        net_returns,
        cast(pd.Series, simulation_result["equity_curve"]),
        cast(pd.Series, simulation_result["drawdown_series"]),
        {
            **execution_metadata,
            "fee_assumption": config.fee_assumption,
            "slippage_assumption": config.slippage_assumption,
        },
    )

    validation_config = config.validation_configuration
    split_ratio = float(validation_config.get("split_ratio", 0.6))
    partitions = InSampleOutOfSampleValidator().split(backtest_result, split_ratio)
    long_window = int(config.strategy_parameters.get("long_window", 50))
    window_defaults = _window_defaults(backtest_result, long_window)
    walk_forward_results = WalkForwardValidator().construct_windows(
        backtest_result,
        int(validation_config.get("train_size", window_defaults["train_size"])),
        int(validation_config.get("evaluation_size", window_defaults["evaluation_size"])),
        int(validation_config.get("step_size", window_defaults["step_size"])),
    )
    walk_forward_rows = _walk_forward_performance(backtest_result, walk_forward_results)
    sensitivity_results = ParameterSensitivityAnalyzer().analyze(
        backtest_result,
        cast(dict[str, list[Any]], validation_config.get("parameter_grid", {"configured": [True]})),
    )
    bootstrap_results = BootstrapAnalyzer().run(
        backtest_result,
        config.random_seed,
        int(validation_config.get("n_samples", 1000)),
    )
    monte_carlo_results = MonteCarloReshuffler().reshuffle(
        backtest_result,
        config.random_seed,
        int(validation_config.get("n_paths", 1000)),
    )

    is_partition = partitions["in_sample"]
    oos_partition = partitions["out_of_sample"]
    is_net_return = float((1 + is_partition.net_returns).prod()) - 1.0  # type: ignore[arg-type]
    oos_net_return = float((1 + oos_partition.net_returns).prod()) - 1.0  # type: ignore[arg-type]

    result_metric = ReturnMetrics().sharpe_ratio
    in_sample_metrics = {
        "sharpe": _safe_metric(result_metric, is_partition),
        "net_return": is_net_return,
        "max_drawdown": DrawdownMetrics().maximum_drawdown(is_partition),
    }
    out_of_sample_metrics = {
        "sharpe": _safe_metric(result_metric, oos_partition),
        "net_return": oos_net_return,
        "max_drawdown": DrawdownMetrics().maximum_drawdown(oos_partition),
    }
    validation_result = ValidationResultAggregator().aggregate(
        in_sample_metrics,
        out_of_sample_metrics,
        walk_forward_results,
        sensitivity_results,
        bootstrap_results,
        monte_carlo_results,
    )

    gross_result = replace(backtest_result, net_returns=backtest_result.gross_returns.copy())
    gross_cumulative = float((1 + backtest_result.gross_returns).prod()) - 1.0  # type: ignore[arg-type]
    net_cumulative = float((1 + backtest_result.net_returns).prod()) - 1.0  # type: ignore[arg-type]
    max_dd = DrawdownMetrics().maximum_drawdown(backtest_result)
    return_metrics = {
        "gross_sharpe": _safe_metric(result_metric, gross_result),
        "net_sharpe": _safe_metric(result_metric, backtest_result),
        "gross_cumulative": gross_cumulative,
        "net_cumulative": net_cumulative,
        "net_return": net_cumulative,
    }
    drawdown_metrics = {
        "maximum_drawdown": max_dd,
        "average_drawdown": DrawdownMetrics().average_drawdown(backtest_result),
        "max_drawdown": max_dd,
    }
    metrics = MetricsAggregator().aggregate(return_metrics, drawdown_metrics, {})

    sw = int(config.strategy_parameters.get("short_window", 10))
    lw = int(config.strategy_parameters.get("long_window", 50))
    sensitivity_grid_rows = _sensitivity_sweep(market_data, config, sw, lw)
    fragility_summary = FragilityEvaluator().evaluate(
        validation_result, metrics, sensitivity_grid_rows
    )
    renderer = PlotlyRenderer()
    charts = [
        renderer.render_equity_curve(backtest_result),
        renderer.render_drawdown(backtest_result),
    ]
    return AuditReportBuilder().build(
        backtest_result,
        validation_result,
        metrics,
        fragility_summary,
        charts,
        sensitivity_grid_rows,
        walk_forward_rows,
    )


def _sensitivity_sweep(
    market_data: MarketData,
    config: AuditConfig,
    base_sw: int,
    base_lw: int,
) -> list[dict[str, Any]]:
    """Run a minimal pipeline for each (short, long) combination around the base parameters.

    Args:
        market_data: Canonical market data shared across all sweep runs.
        config: Audit configuration whose cost and execution assumptions are reused.
        base_sw: Configured short window to vary around.
        base_lw: Configured long window to vary around.

    Returns:
        Rows of ``{"Short": int, "Long": int, "Sharpe": float}`` for valid pairs.

    Raises:
        Nothing — failed combinations produce NaN Sharpe and are included.

    Invariants:
        Does not alter validation, bootstrap, or Monte Carlo state.
        Only signal→simulate→cost→sharpe is executed per combination.
    """
    step_s = max(2, base_sw // 2)
    step_l = max(5, base_lw // 5)
    short_candidates = sorted(
        {max(2, base_sw - step_s), base_sw, base_sw + step_s, base_sw + step_s * 2}
    )
    long_candidates = sorted(
        {max(base_sw + 1, base_lw - step_l * 2), base_lw - step_l, base_lw, base_lw + step_l}
    )
    cost_cfg = _cost_config(config)
    result_metric = ReturnMetrics().sharpe_ratio
    rows = []
    for sw in short_candidates:
        for lw in long_candidates:
            if sw >= lw:
                continue
            params = {**config.strategy_parameters, "short_window": sw, "long_window": lw}
            try:
                sigs = MovingAverageCrossoverStrategy().generate_signals(market_data, params)
                sized = PositionSizer().size_positions(
                    sigs, cast("dict[str, Any]", config.position_sizing)
                )
                executed, exec_meta = ExecutionModel().apply_execution_timing(
                    sized, market_data.open, config.execution_assumptions
                )
                sim = PortfolioSimulator().simulate(market_data, executed, {"initial_equity": 1.0})
                gross = cast(pd.Series, sim["gross_returns"])
                trades = cast(pd.DataFrame, sim["trades"])
                net = CostModel().apply_costs(gross, trades, cost_cfg)
                br = BacktestResultBuilder().build(
                    cast(pd.Series, sim["positions"]),
                    trades,
                    gross,
                    net,
                    cast(pd.Series, sim["equity_curve"]),
                    cast(pd.Series, sim["drawdown_series"]),
                    exec_meta,
                )
                sharpe = _safe_metric(result_metric, br)
            except (ValueError, KeyError):
                sharpe = float("nan")
            rows.append({"Short": sw, "Long": lw, "Sharpe": round(sharpe, 3)})
    return rows


def _walk_forward_performance(
    backtest_result: BacktestResult,
    windows: list[dict[str, int]],
) -> list[dict[str, Any]]:
    """Compute eval-period Sharpe for each walk-forward window.

    Args:
        backtest_result: Full-period canonical engine output to slice from.
        windows: Index-based windows produced by ``WalkForwardValidator``.

    Returns:
        Rows of ``{"Fold": int, "Eval period": str, "Sharpe": float}``.

    Raises:
        Nothing — windows with fewer than two observations are skipped.

    Invariants:
        Uses existing net_returns only; does not refit or re-optimize parameters.
    """
    result_metric = ReturnMetrics().sharpe_ratio
    rows = []
    for i, window in enumerate(windows, start=1):
        eval_slice = slice(window["evaluation_start"], window["evaluation_end"])
        eval_net = backtest_result.net_returns.iloc[eval_slice]
        eval_gross = backtest_result.gross_returns.iloc[eval_slice].copy()
        if len(eval_net) < 2:
            continue
        eval_br = replace(backtest_result, net_returns=eval_net, gross_returns=eval_gross)
        sharpe = _safe_metric(result_metric, eval_br)
        start = str(eval_net.index[0])[:10]
        end = str(eval_net.index[-1])[:10]
        rows.append({"Fold": i, "Eval period": f"{start} → {end}", "Sharpe": round(sharpe, 3)})
    return rows


def _cost_config(config: AuditConfig) -> dict[str, Any]:
    """Collect explicit cost assumptions for the engine cost model.

    Args:
        config: Canonical audit configuration containing fee and slippage
            assumptions.

    Returns:
        A flat cost-configuration dictionary for ``CostModel``.

    Raises:
        ValueError: Propagated later by ``CostModel`` if required cost keys are
            absent or malformed.

    Invariants:
        This helper only maps configuration data; it does not apply costs or
        calculate returns.
    """
    cost_config: dict[str, Any] = {}
    if isinstance(config.fee_assumption, dict):
        cost_config.update(config.fee_assumption)
    if isinstance(config.slippage_assumption, dict):
        cost_config.update(config.slippage_assumption)
    return cost_config


def _safe_metric(metric: Any, backtest_result: BacktestResult) -> float:
    """Invoke a metrics-layer callable with deterministic MVP defaults.

    Args:
        metric: Metrics-layer callable to execute.
        backtest_result: Canonical engine output consumed by the metric.

    Returns:
        Numeric metric value, or a deterministic fallback when the metric
        component rejects an undersized MVP fixture.

    Raises:
        TypeError: If the callable signature is incompatible.

    Invariants:
        This helper delegates to the metrics layer and contains no metric
        formula.
    """
    try:
        return float(metric(backtest_result, 0.0, 252.0))
    except ValueError:
        return float("nan")


def _window_defaults(backtest_result: BacktestResult, long_window: int = 50) -> dict[str, int]:
    """Choose walk-forward window sizes following Lopez de Prado (AFML, ch. 12).

    Args:
        backtest_result: Canonical engine output whose observation count bounds
            the walk-forward configuration.
        long_window: Strategy's longest lookback period; sets the minimum
            meaningful train size (must see the signal at least three times).

    Returns:
        Default training, evaluation, and step sizes that fit the available
        observations.

    Raises:
        ValueError: Propagated later by ``WalkForwardValidator`` if the
            available observations cannot support a valid window.

    Invariants:
        This helper chooses window sizes only; it does not evaluate strategy
        performance or calculate validation metrics.
    """
    observation_count = len(backtest_result.net_returns)
    train_size = max(long_window * 3, observation_count // 5)
    evaluation_size = max(long_window, train_size // 3)
    train_size = min(train_size, observation_count - evaluation_size)
    train_size = max(1, train_size)
    evaluation_size = max(1, min(evaluation_size, observation_count - train_size))
    return {
        "train_size": train_size,
        "evaluation_size": evaluation_size,
        "step_size": evaluation_size,
    }
