"""Canonical domain contracts for Backtest Auditor.

This module is the single public definition point for cross-layer contract
types. Phase 1 intentionally records the contractual surface only; concrete
dataclass implementations are added after human review.
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

__all__ = [
    "AuditConfig",
    "AuditReport",
    "BacktestResult",
    "MarketData",
    "ValidationResult",
]


@dataclass(kw_only=True)
class AuditConfig:
    """Full reproducible audit run configuration.

    Args:
        strategy_name: Stable strategy identifier selected by the caller. It
            must name a supported reference strategy or compatible strategy
            adapter and must not be empty.
        strategy_parameters: Strategy-specific parameter mapping. Values must
            be explicit inputs to the audit; validation and sensitivity layers
            must not mutate this mapping to imply optimized parameters.
        fee_assumption: Explicit fee model or fee configuration consumed by the
            engine. A zero-cost audit must be represented explicitly rather
            than inferred from a missing value.
        slippage_assumption: Explicit slippage model or slippage configuration
            consumed by the engine. A zero-slippage audit must be represented
            explicitly rather than inferred from a missing value.
        position_sizing: Position-sizing rule or configuration consumed by the
            engine. It must be sufficient to reproduce generated positions.
        execution_assumptions: Signal-to-execution timing and any other
            execution assumptions. Default timing is signal at bar T close and
            execution at bar T+1 open unless this field explicitly records a
            different assumption.
        validation_configuration: Configuration for IS/OOS, walk-forward,
            sensitivity, bootstrap, and Monte Carlo procedures. It must
            preserve validation independence and must not describe an
            optimization search.
        random_seed: Explicit integer seed used by stochastic validation
            procedures. The same inputs and the same seed must reproduce the
            same stochastic outputs.

    Raises:
        ValueError: If required assumptions are omitted, random_seed is absent,
            or a configuration implies silent zero-cost execution, parameter
            optimization, look-ahead execution, or merged IS/OOS evaluation.

    Invariants:
        All assumptions needed to reproduce a run are explicit.
        random_seed is present and typed as int.
        Cost assumptions are configuration inputs for the engine only.
        Sensitivity configuration measures fragility around fixed parameters
        and never selects, ranks, or recommends parameter sets.
    """

    strategy_name: str
    strategy_parameters: dict[str, Any] = field(default_factory=dict)
    fee_assumption: Any
    slippage_assumption: Any
    position_sizing: Any
    execution_assumptions: dict[str, Any] = field(default_factory=dict)
    validation_configuration: dict[str, Any] = field(default_factory=dict)
    random_seed: int


@dataclass(kw_only=True)
class MarketData:
    """Canonical OHLCV data emitted by the data layer.

    Args:
        timestamp: Chronologically ordered timestamps with no duplicates. Each
            timestamp identifies the bar represented by the corresponding OHLCV
            values.
        open: Opening prices aligned one-to-one with timestamp.
        high: High prices aligned one-to-one with timestamp. Each value must be
            structurally valid relative to open, low, and close.
        low: Low prices aligned one-to-one with timestamp. Each value must be
            structurally valid relative to open, high, and close.
        close: Closing prices aligned one-to-one with timestamp.
        volume: Volume observations aligned one-to-one with timestamp.

    Raises:
        ValueError: If timestamps are unordered or duplicated, field lengths
            differ, required OHLCV values are missing, or OHLCV relationships
            are structurally invalid.

    Invariants:
        MarketData is validated before it enters the engine.
        All series have identical length and index alignment.
        Rows are chronological and contain no look-ahead-derived values.
        This type contains market data only; it carries no signals, costs,
        metrics, validation results, or reporting conclusions.
    """

    timestamp: pd.Series
    open: pd.Series
    high: pd.Series
    low: pd.Series
    close: pd.Series
    volume: pd.Series


@dataclass(kw_only=True)
class BacktestResult:
    """Canonical output of the backtest engine.

    Args:
        positions: Position series produced by the engine after applying the
            configured strategy intent, position sizing, and execution model.
        trades: Normalized trade records emitted by the engine.
        gross_returns: Period returns before fees and slippage. This field
            represents idealized performance and must be distinguishable from
            cost-adjusted performance.
        net_returns: Period returns after explicit engine-applied fees and
            slippage. This field must never reference the same object as
            gross_returns.
        equity_curve: Equity series derived from the engine output and aligned
            with the return series.
        drawdown_series: Drawdown series aligned with equity_curve.
        execution_metadata: Metadata sufficient to audit fee assumptions,
            slippage assumptions, position sizing, signal-to-execution timing,
            and other execution assumptions used by the engine.

    Raises:
        ValueError: If gross_returns or net_returns are missing, if they are the
            same object, if cost assumptions are not traceable in
            execution_metadata, or if execution timing is absent or ambiguous.

    Invariants:
        Gross and net returns are separate first-class fields.
        gross_returns is not net_returns.
        Fees and slippage are applied by the engine, not metrics or reporting.
        Execution assumptions are explicit enough to detect look-ahead bias.
    """

    positions: pd.Series
    trades: pd.DataFrame
    gross_returns: pd.Series
    net_returns: pd.Series
    equity_curve: pd.Series
    drawdown_series: pd.Series
    execution_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.gross_returns is self.net_returns:
            raise ValueError("gross_returns and net_returns must not be the same object")


@dataclass(kw_only=True)
class ValidationResult:
    """Canonical output of the validation layer.

    Args:
        in_sample_metrics: Metrics for the in-sample segment only. These values
            must remain separate from out-of-sample metrics.
        out_of_sample_metrics: Metrics for the out-of-sample segment only.
            These values must remain separate from in-sample metrics.
        walk_forward_results: Sequential walk-forward window results. Evaluation
            windows must be non-overlapping and individually attributable.
        sensitivity_results: Parameter-sensitivity diagnostics around a fixed
            parameter point. Results must describe stability or fragility and
            must not recommend a best parameter set.
        bootstrap_results: Bootstrap outputs, including enough metadata to
            reproduce stochastic procedures from the configured random_seed.
        monte_carlo_results: Monte Carlo reshuffling outputs, including enough
            metadata to reproduce stochastic procedures from the configured
            random_seed.

    Raises:
        ValueError: If in-sample and out-of-sample evidence is merged, if
            walk-forward evaluation windows overlap, if sensitivity results
            rank or select parameters, or if stochastic result metadata omits
            the random seed.

    Invariants:
        in_sample_metrics and out_of_sample_metrics are separate fields.
        No merged headline metric may replace IS/OOS evidence.
        Each fragility-relevant conclusion remains traceable to the validation
        procedure that produced it.
        Stochastic validation outputs preserve reproducibility metadata.
    """

    in_sample_metrics: dict[str, Any] = field(default_factory=dict)
    out_of_sample_metrics: dict[str, Any] = field(default_factory=dict)
    walk_forward_results: list[dict[str, Any]] = field(default_factory=list)
    sensitivity_results: dict[str, Any] = field(default_factory=dict)
    bootstrap_results: dict[str, Any] = field(default_factory=dict)
    monte_carlo_results: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class AuditReport:
    """Final report model assembled by the reporting layer.

    Args:
        baseline_metrics: Metrics describing baseline or gross performance.
            These values must be traceable to analytical outputs and must not be
            recalculated by reporting.
        cost_adjusted_metrics: Metrics describing net, cost-adjusted
            performance. These values must preserve the distinction between
            gross and net results.
        out_of_sample_metrics: Out-of-sample evidence reported separately from
            baseline and in-sample evidence.
        walk_forward_summary: Reporting summary of walk-forward outcomes with
            enough attribution to inspected windows.
        parameter_sensitivity_summary: Reporting summary of sensitivity
            diagnostics. It must describe fragility and must not recommend an
            optimized parameter set.
        simulated_drawdown_summary: Reporting summary of bootstrap or Monte
            Carlo drawdown evidence.
        regime_summary: Reporting summary of regime-oriented performance
            evidence.
        fragility_summary: Descriptive conclusions tied directly to observed
            evidence such as cost deterioration, OOS weakness, unstable
            walk-forward results, narrow parameter robustness, simulated
            drawdown risk, or regime dependence.
        verdict: Final verdict string derived by the reporting layer from
            explicit evidence in the report. It has no default value because a
            default would imply an unsupported conclusion.
        charts: Presentation-ready chart objects or chart view models consumed
            by the UI.

    Raises:
        ValueError: If verdict is omitted, defaulted, or unsupported by report
            evidence; if reporting recalculates analytical results; or if IS/OOS
            or gross/net distinctions are collapsed.

    Invariants:
        verdict is derived from explicit evidence and has no conclusion-implying
        default.
        Reporting consumes analytical outputs only.
        Fragility statements are descriptive, evidence-based, and do not imply
        guaranteed future profitability.
        Gross/net and IS/OOS evidence remain distinguishable in the report.
    """

    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    cost_adjusted_metrics: dict[str, Any] = field(default_factory=dict)
    out_of_sample_metrics: dict[str, Any] = field(default_factory=dict)
    walk_forward_summary: dict[str, Any] = field(default_factory=dict)
    parameter_sensitivity_summary: dict[str, Any] = field(default_factory=dict)
    simulated_drawdown_summary: dict[str, Any] = field(default_factory=dict)
    regime_summary: dict[str, Any] = field(default_factory=dict)
    fragility_summary: list[str] = field(default_factory=list)
    verdict: str
    charts: list[Any] = field(default_factory=list)
