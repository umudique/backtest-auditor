"""Top-level Streamlit application contract."""

from src.orchestrator import run_audit


def run_app() -> None:
    """Run the Streamlit UI entry point.

    Args:
        None.

    Returns:
        None.

    Raises:
        NotImplementedError: Until Phase 2 supplies Streamlit interaction
            wiring.
        ValueError: If user-provided inputs fail surface-level UI validation.

    Invariants:
        The UI is a thin presentation and orchestration boundary.
        It calls ``validate_upload``, ``build_config``,
        ``src.orchestrator.run_audit``, and ``render_report``.
        It does not import or call engine, validation, metrics, or data pipeline
        stages directly and contains no financial calculations.
        Session state, when introduced, may hold configuration and
        ``AuditReport`` objects only.
    """
    print(f"Backtest Auditor UI ready: {run_audit.__name__}")
