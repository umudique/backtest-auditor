"""Verdict-first report presentation for the Streamlit UI boundary."""

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.contracts import AuditConfig, AuditReport

_VERDICT_COLORS = {
    "ROBUST": "#1e7a48",
    "FRAGILE": "#c4890a",
    "FAIL": "#b03535",
}


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
        Impact (gross − net cumulative) and IS/OOS Delta (OOS − IS per metric).
    """
    if report.verdict not in {"ROBUST", "FRAGILE", "FAIL"}:
        raise ValueError("AuditReport.verdict must be ROBUST, FRAGILE, or FAIL")

    _render_verdict_layer(report)

    _spacer, _btn_col = st.columns([10, 1])
    with _btn_col:
        if st.button("← New Audit", key="reset_audit", use_container_width=True):
            st.session_state.pop("audit_report", None)
            st.session_state.pop("audit_config", None)
            st.rerun()

    left, right = st.columns(2)

    with left:
        _render_charts(report)
        _render_fragility_layer(report)

    with right:
        _render_generalization_layer(report)
        _render_stress_layer(report)

    with st.expander("Assumptions", expanded=False):
        _render_inputs_layer(config)


def _render_verdict_layer(report: AuditReport) -> None:
    """Render the verdict chip and five KPI metrics in a single row."""
    color = _VERDICT_COLORS.get(report.verdict, "#555")
    cost_impact = _cost_impact(report)

    c0, c1, c2, c3, c4, c5 = st.columns(6)
    c0.markdown(
        f"""<div style="margin-top:0.25rem">
          <div class="metric-label" style="color:#888; margin-bottom:0.15rem">BACKTEST VERDICT</div>
          <div class="metric-value bold" style="color:{color}; line-height:1.1">
            {report.verdict}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    c1.metric("Net Return", _fmt_pct(report.cost_adjusted_metrics.get("net_return")))
    c2.metric("OOS Sharpe", _fmt_num(report.out_of_sample_metrics.get("sharpe")))
    c3.metric("Max Drawdown", _fmt_pct(report.baseline_metrics.get("maximum_drawdown")))
    c4.metric("Cost Drag", _fmt_pp(cost_impact))
    c5.metric(
        "Median Monte Carlo Drawdown",
        _fmt_pct(report.simulated_drawdown_summary.get("median_max_drawdown")),
    )


_CHART_HEADERS = [
    (
        "**Equity Curve**",
        "Gross vs. net cumulative return over the full backtest period. "
        "The gap between lines reflects the total cost drag.",
    ),
    (
        "**Drawdown**",
        "Percentage decline from the rolling equity peak. "
        "Prolonged or deep troughs indicate periods of elevated risk.",
    ),
]


def _render_charts(report: AuditReport) -> None:
    """Render gross/net equity and drawdown charts, height-constrained."""
    for i, chart in enumerate(report.charts):
        if i < len(_CHART_HEADERS):
            title, caption = _CHART_HEADERS[i]
            st.markdown(title)
            st.caption(caption)
        try:
            chart.update_layout(
                height=290,
                margin={"t": 10, "b": 24, "l": 48, "r": 16},
                legend={"orientation": "h", "y": -0.15},
            )
            st.plotly_chart(chart, use_container_width=True)
        except Exception:
            st.write(chart)


def _render_generalization_layer(report: AuditReport) -> None:
    """Render IS/OOS evidence and walk-forward evidence without recomputation."""
    st.markdown("**Generalization**")
    st.caption("Large negative OOS Delta is a sign of overfitting.")

    rows = [
        {
            "Metric": label,
            "In-sample": _fmt_num3(report.baseline_metrics.get(key)),
            "Out-of-sample": _fmt_num3(report.out_of_sample_metrics.get(key)),
            "Delta": _fmt_num3(_metric_delta(report, key)),
        }
        for key, label in (
            ("sharpe", "Sharpe"),
            ("net_return", "Net Return"),
            ("max_drawdown", "Max Drawdown"),
        )
    ]
    st.table(rows)

    folds = report.walk_forward_summary.get("folds") if report.walk_forward_summary else None
    if isinstance(folds, list) and folds:
        st.markdown("**Walk-forward Analysis**")
        st.caption(
            "Does the strategy perform consistently across different time periods? "
            "Declining Sharpe across folds indicates time-period dependence. "
            "Fold Sharpe is annualized over short eval windows — values are not "
            "directly comparable to the full-period OOS Sharpe."
        )
        _render_wfa_chart(folds)


def _fragility_chip(finding: str, report: AuditReport) -> tuple[str, str]:
    """Derive (category_label, human_description) from a finding string."""
    if "cost deterioration" in finding:
        gross = report.baseline_metrics.get("gross_sharpe")
        net = report.cost_adjusted_metrics.get("net_sharpe")
        if isinstance(gross, (int, float)) and isinstance(net, (int, float)):
            if gross <= 0:
                desc = "Fees amplify an already losing strategy"
            elif net < 0:
                desc = "Fees turn a gross-profitable strategy into a net loss"
            else:
                pct = (gross - net) / abs(gross) * 100
                desc = f"Fees consume {pct:.0f}% of gross return"
        else:
            desc = "Transaction costs significantly reduce performance"
        return "COST", desc

    if "out-of-sample" in finding:
        is_s = report.baseline_metrics.get("sharpe")
        oos_s = report.out_of_sample_metrics.get("sharpe")
        if isinstance(is_s, (int, float)) and isinstance(oos_s, (int, float)):
            diff = oos_s - is_s
            desc = f"Sharpe deteriorated {abs(diff):.3f} (IS {is_s:.3f} → OOS {oos_s:.3f})"
        else:
            desc = "Performance degrades significantly on unseen data"
        return "OOS", desc

    if "narrow profitable plateau" in finding:
        rows = report.sensitivity_grid_rows
        if rows:
            total = len(rows)
            profitable = sum(
                1 for r in rows if isinstance(r.get("Sharpe"), (int, float)) and r["Sharpe"] > 0
            )
            desc = (
                f"No profitable parameter combination found across {total} tested"
                if profitable == 0
                else f"Profitable in {profitable} of {total} parameter combinations"
            )
        else:
            desc = "No profitable parameter region found"
        return "SENSITIVITY", desc

    return "RISK", finding


def _render_fragility_chips(report: AuditReport) -> None:
    """Render each fragility finding as a clean diagnostic row."""
    rows_html = ""
    for finding in report.fragility_summary:
        label, desc = _fragility_chip(finding, report)
        rows_html += (
            f"<tr>"
            f"<td style=\"font-family:'IBM Plex Mono',monospace; font-size:0.68rem;"
            f"font-weight:600; color:#9b2020; letter-spacing:0.06em;"
            f'padding:0.3rem 0.8rem 0.3rem 0; white-space:nowrap; vertical-align:top">{label}</td>'
            f'<td style="font-size:0.8rem; color:#444; padding:0.3rem 0;'
            f'line-height:1.4; vertical-align:top">{desc}</td>'
            f"</tr>"
        )
    st.markdown(
        f'<table style="width:100%; border-collapse:collapse;'
        f"border-top:1px solid #ddd; border-bottom:1px solid #ddd;"
        f'margin-bottom:0.6rem">{rows_html}</table>',
        unsafe_allow_html=True,
    )


def _render_fragility_layer(report: AuditReport) -> None:
    """Render findings as human-readable chips and sensitivity heatmap."""
    st.markdown("**Parameter Sensitivity**")
    st.caption(
        "Does the result hold when parameters shift slightly? "
        "A narrow profitable region indicates the result is coincidental."
    )
    st.write("Sensitivity Heatmap")
    if report.sensitivity_grid_rows:
        _render_sensitivity_heatmap(report.sensitivity_grid_rows)
    else:
        st.caption("Sensitivity Heatmap not yet computed.")
    if report.fragility_summary:
        _render_fragility_chips(report)


def _render_wfa_chart(folds: list[dict[str, Any]]) -> None:
    """Render a horizontal bar chart of per-fold OOS Sharpe ratios."""
    df = pd.DataFrame(folds)
    colors = [
        _VERDICT_COLORS["ROBUST"] if s >= 0 else _VERDICT_COLORS["FAIL"] for s in df["Sharpe"]
    ]
    fig = go.Figure(
        go.Bar(
            x=df["Sharpe"],
            y=df["Eval period"],
            orientation="h",
            marker_color=colors,
            text=[f"{s:.2f}" for s in df["Sharpe"]],
            textposition="outside",
        )
    )
    fig.add_vline(x=0, line_width=1, line_color="#aaa", line_dash="dot")
    fig.update_layout(
        height=max(160, len(folds) * 36 + 40),
        margin={"t": 10, "b": 30, "l": 10, "r": 50},
        xaxis_title="Sharpe",
        yaxis={"autorange": "reversed"},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_sensitivity_heatmap(rows: list[dict[str, Any]]) -> None:
    """Render a Sharpe heatmap pivoted on Short × Long window."""
    try:
        df = pd.DataFrame(rows)
        pivot = df.pivot(index="Short", columns="Long", values="Sharpe")
        z = pivot.values.tolist()
        text = [[f"{v:.2f}" for v in row] for row in pivot.values]
        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=[str(c) for c in pivot.columns],
                y=[str(r) for r in pivot.index],
                colorscale="RdYlGn",
                zmid=0,
                text=text,
                texttemplate="%{text}",
                showscale=True,
                colorbar={"thickness": 12, "len": 0.8},
            )
        )
        fig.update_layout(
            height=max(200, len(pivot.index) * 40 + 80),
            margin={"t": 10, "b": 50, "l": 60, "r": 20},
            xaxis_title="Long window",
            yaxis_title="Short window",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    except (KeyError, ValueError):
        st.table(rows)


def _render_stress_layer(report: AuditReport) -> None:
    """Render Monte Carlo drawdown distribution histogram with reference lines."""
    st.markdown("**Stress Test**")
    st.caption(
        "How does the strategy behave across randomly resampled histories? "
        "Wide outcome distributions indicate sensitivity to path order. "
        "Monte Carlo reshuffling destroys serial correlation, so simulated drawdowns "
        "tend to exceed the actual — actual falling outside the band is expected."
    )
    summary = report.simulated_drawdown_summary
    dist = summary.get("drawdown_distribution") if summary else None
    if not dist:
        st.caption("Monte Carlo distributions not yet computed.")
        return

    median_dd = summary.get("median_max_drawdown")
    p05 = summary.get("p05")
    p95 = summary.get("p95")
    actual_dd = report.baseline_metrics.get("maximum_drawdown")
    n_paths = summary.get("n_paths", len(dist))

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=dist,
            nbinsx=50,
            marker_color="#6b8cae",
            opacity=0.85,
            name=f"{n_paths} MC paths",
        )
    )
    if p05 is not None and p95 is not None:
        fig.add_vrect(
            x0=p05,
            x1=p95,
            fillcolor="rgba(107,140,174,0.12)",
            line_width=0,
            annotation_text="P5–P95",
            annotation_position="top left",
            annotation_font_size=10,
        )
    if median_dd is not None:
        fig.add_vline(
            x=median_dd,
            line_width=1.5,
            line_dash="dash",
            line_color="#888",
            annotation_text=f"Median {_fmt_pct(median_dd)}",
            annotation_position="top right",
            annotation_font_size=10,
        )
    if actual_dd is not None:
        fig.add_vline(
            x=actual_dd,
            line_width=2,
            line_color=_VERDICT_COLORS["FAIL"],
            annotation_text=f"Actual {_fmt_pct(actual_dd)}",
            annotation_position="top left",
            annotation_font_size=10,
        )
    fig.update_layout(
        height=240,
        margin={"t": 30, "b": 40, "l": 10, "r": 10},
        xaxis_title="Max Drawdown",
        xaxis_tickformat=".0%",
        yaxis_title="Paths",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_inputs_layer(config: AuditConfig | None) -> None:
    """Render stored audit assumptions without deriving analytical values."""
    if config is None:
        st.caption("Configuration not available.")
        return

    fee = config.fee_assumption
    slip = config.slippage_assumption
    rows = [
        ("Strategy", config.strategy_name),
        ("Short window", config.strategy_parameters.get("short_window", "—")),
        ("Long window", config.strategy_parameters.get("long_window", "—")),
        ("Fee rate", _fmt_pct(fee.get("fee_rate") if isinstance(fee, dict) else None)),
        ("Slippage rate", _fmt_pct(slip.get("slippage_rate") if isinstance(slip, dict) else None)),
        ("Random seed", config.random_seed),
    ]
    for label, value in rows:
        st.write(f"**{label}:** {value}")


def _cost_impact(report: AuditReport) -> Any:
    """Return the allowed Cost Impact subtraction when both values exist."""
    gross = report.baseline_metrics.get("gross_cumulative")
    net = report.cost_adjusted_metrics.get("net_cumulative")
    if isinstance(gross, int | float) and isinstance(net, int | float):
        return gross - net
    return None


def _metric_delta(report: AuditReport, metric: str) -> Any:
    """Return the approved OOS minus IS display delta when both values exist."""
    is_val = report.baseline_metrics.get(metric)
    oos_val = report.out_of_sample_metrics.get(metric)
    if isinstance(is_val, int | float) and isinstance(oos_val, int | float):
        return oos_val - is_val
    return None


def _fmt_pct(val: Any) -> str:
    if val is None:
        return "—"
    return f"{val:+.1%}"


def _fmt_pp(val: Any) -> str:
    if val is None:
        return "—"
    return f"{val:+.1%} pp"


def _fmt_num(val: Any) -> str:
    if val is None:
        return "—"
    return f"{float(val):.2f}"


def _fmt_num3(val: Any) -> str:
    if val is None:
        return "—"
    return f"{float(val):.3f}"


def _array_len(obj: Any) -> int | None:
    try:
        return len(obj)
    except TypeError:
        return None
