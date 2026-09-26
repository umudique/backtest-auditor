"""Verdict-first report presentation for the Streamlit UI boundary."""

from typing import Any

import streamlit as st

from src.contracts import AuditConfig, AuditReport


def render_report(report: AuditReport, config: AuditConfig | None = None) -> None:
    """Render an audit report in the verdict-first architecture order.

    Args:
        report: Canonical report produced by ``src.orchestrator.run_audit``.
        config: Optional canonical audit configuration stored in session state
            and displayed only in the final assumptions layer.

    Returns:
        None.

    Raises:
        ValueError: If ``report.verdict`` is not one of the canonical verdict
            labels.

    Invariants:
        The verdict banner is rendered before analytical evidence.
        Render order is verdict, performance reality, generalization,
        fragility, stress, and inputs.
        The UI displays ``AuditReport`` fields as supplied by reporting and
        never imports engine, validation, metrics, or data modules.
        The only arithmetic allowed in this presentation boundary is Cost
        Impact, calculated as already-produced gross cumulative minus already-
        produced net cumulative, and IS/OOS Delta, calculated as already-
        produced out-of-sample metric minus already-produced in-sample metric.
    """
    if report.verdict not in {"ROBUST", "FRAGILE", "FAIL"}:
        raise ValueError("AuditReport.verdict must be ROBUST, FRAGILE, or FAIL")

    _render_verdict_layer(report)
    _render_performance_reality_layer(report)
    _render_generalization_layer(report)

    with st.expander("Layer 4 - Fragility", expanded=False):
        _render_fragility_layer(report)

    with st.expander("Layer 5 - Stress", expanded=False):
        _render_stress_layer(report)

    with st.expander("Layer 6 - Inputs and Assumptions", expanded=False):
        _render_inputs_layer(config)


def _render_verdict_layer(report: AuditReport) -> None:
    """Render the top verdict banner and five required KPI labels."""
    st.markdown(f"BACKTEST VERDICT: {report.verdict}")
    st.write(
        {
            "Net Return": report.cost_adjusted_metrics.get("net_return"),
            "OOS Sharpe": report.out_of_sample_metrics.get("sharpe"),
            "Max Drawdown": report.baseline_metrics.get("max_drawdown"),
            "Cost Impact": _cost_impact(report),
            "Monte Carlo DD": report.simulated_drawdown_summary.get("median_max_drawdown"),
        }
    )


def _render_performance_reality_layer(report: AuditReport) -> None:
    """Render gross/net equity and drawdown chart objects supplied by reporting."""
    st.subheader("Layer 2 - Performance Reality")
    st.write("Gross return")
    st.write("Net return")
    for chart in report.charts:
        st.write(chart)


def _render_generalization_layer(report: AuditReport) -> None:
    """Render IS/OOS evidence and walk-forward evidence without recomputation."""
    st.subheader("Layer 3 - Generalization")
    st.write(
        [
            {
                "Metric": metric,
                "In-sample": report.baseline_metrics.get(metric),
                "Out-of-sample": report.out_of_sample_metrics.get(metric),
                "Delta": _metric_delta(report, metric),
            }
            for metric in ("sharpe", "net_return", "max_drawdown")
        ]
    )

    st.write("Walk-forward")
    if report.walk_forward_summary:
        st.write(report.walk_forward_summary)
    else:
        st.info("Walk-forward results not yet computed")


def _render_fragility_layer(report: AuditReport) -> None:
    """Render sensitivity heatmap data or the labelled placeholder."""
    st.write("Sensitivity Heatmap")
    if report.sensitivity_grid_rows:
        st.write(report.sensitivity_grid_rows)
    else:
        st.info("Sensitivity Heatmap not yet computed")

    for conclusion in report.fragility_summary:
        st.write(conclusion)


def _render_stress_layer(report: AuditReport) -> None:
    """Render stress distribution fields supplied by reporting."""
    st.write("Monte Carlo distributions")
    if report.simulated_drawdown_summary:
        st.write(report.simulated_drawdown_summary)
    else:
        st.info("Monte Carlo distributions not yet computed")


def _render_inputs_layer(config: AuditConfig | None) -> None:
    """Render stored audit assumptions without deriving analytical values."""
    if config is None:
        st.info("AuditConfig not available in session state")
        return

    st.write(
        {
            "Strategy": config.strategy_name,
            "Strategy parameters": config.strategy_parameters,
            "Fee assumption": config.fee_assumption,
            "Slippage assumption": config.slippage_assumption,
            "Position sizing": config.position_sizing,
            "Execution assumptions": config.execution_assumptions,
            "Validation configuration": config.validation_configuration,
            "Random seed": config.random_seed,
        }
    )


def _cost_impact(report: AuditReport) -> Any:
    """Return the allowed Cost Impact subtraction when both values exist."""
    gross_cumulative = report.baseline_metrics.get("gross_cumulative")
    net_cumulative = report.cost_adjusted_metrics.get("net_cumulative")
    if isinstance(gross_cumulative, int | float) and isinstance(net_cumulative, int | float):
        return gross_cumulative - net_cumulative
    return None


def _metric_delta(report: AuditReport, metric: str) -> Any:
    """Return the approved OOS minus IS display delta when both values exist."""
    in_sample = report.baseline_metrics.get(metric)
    out_of_sample = report.out_of_sample_metrics.get(metric)
    if isinstance(in_sample, int | float) and isinstance(out_of_sample, int | float):
        return out_of_sample - in_sample
    return None
