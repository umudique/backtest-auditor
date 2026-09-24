"""Display contracts for the Streamlit UI boundary."""

import streamlit as st

from src.contracts import AuditReport


def render_report(report: AuditReport) -> None:
    """Render all fields of an audit report.

    Args:
        report: Canonical audit report produced by ``src.orchestrator`` and the
            reporting layer.

    Returns:
        None.

    Raises:
        NotImplementedError: Until Phase 2 supplies Streamlit display calls.
        ValueError: If the report is missing required presentation fields.

    Invariants:
        Displays ``AuditReport`` fields as produced.
        Does not import or call engine, validation, metrics, data, or reporting
        calculation components.
        Does not recalculate, reformat, reorder, or re-derive analytical
        conclusions.
    """
    st.subheader("Verdict")
    st.write(report.verdict)

    st.subheader("Fragility Summary")
    render_fragility_summary(report.fragility_summary)

    st.subheader("Baseline Metrics")
    st.json(report.baseline_metrics)

    st.subheader("Cost-Adjusted Metrics")
    st.json(report.cost_adjusted_metrics)

    st.subheader("Out-of-Sample Metrics")
    st.json(report.out_of_sample_metrics)

    for chart in report.charts:
        try:
            st.plotly_chart(chart, use_container_width=True)
        except (TypeError, ValueError):
            st.write(chart)


def render_fragility_summary(fragility_summary: list[str]) -> None:
    """Render fragility conclusions in their supplied order.

    Args:
        fragility_summary: Evidence-based conclusions from ``AuditReport``.

    Returns:
        None.

    Raises:
        NotImplementedError: Until Phase 2 supplies Streamlit display calls.
        ValueError: If the fragility summary is malformed for display.

    Invariants:
        Renders conclusions without re-deriving, filtering, sorting, scoring,
        or reordering them.
        Does not inspect metrics or validation outputs directly.
    """
    for conclusion in fragility_summary:
        st.markdown(f"- {conclusion}")
