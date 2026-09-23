"""Display contracts for the Streamlit UI boundary."""

from dataclasses import fields

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
    for report_field in fields(AuditReport):
        print(f"{report_field.name}: {getattr(report, report_field.name)}")


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
        print(conclusion)
