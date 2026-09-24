"""Top-level Streamlit application contract."""

import pathlib
import tempfile

import streamlit as st

from app.display import render_report
from app.inputs import build_config, load_market_data
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
    st.title("Backtest Auditor")

    uploaded = st.file_uploader("Upload market data", type=["csv", "parquet"])
    with st.sidebar.form("audit_configuration"):
        strategy_name = st.text_input("Strategy name", value="moving_average_crossover")
        short_window = st.number_input("Short window", min_value=1, value=2, step=1)
        long_window = st.number_input("Long window", min_value=2, value=3, step=1)
        fee_rate = st.number_input("Fee rate", min_value=0.0, value=0.001, format="%.6f")
        slippage_rate = st.number_input(
            "Slippage rate",
            min_value=0.0,
            value=0.001,
            format="%.6f",
        )
        random_seed = st.number_input("Random seed", min_value=0, value=7, step=1)
        submitted = st.form_submit_button("Run audit")

    if not submitted:
        return

    if uploaded is None:
        st.error("Upload market data before running an audit.")
        return

    suffix = pathlib.Path(uploaded.name).suffix
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_file.write(uploaded.read())
            temporary_path = temporary_file.name

        form_values = {
            "strategy_name": strategy_name,
            "strategy_parameters": {
                "short_window": int(short_window),
                "long_window": int(long_window),
            },
            "fee_assumption": {"fee_rate": float(fee_rate)},
            "slippage_assumption": {"slippage_rate": float(slippage_rate)},
            "position_sizing": {"min_position": -1.0, "max_position": 1.0},
            "execution_assumptions": {"timing": "signal_close_execute_next_open"},
            "validation_configuration": {"split_ratio": 0.6},
            "random_seed": int(random_seed),
        }
        config = build_config(form_values)
        market_data = load_market_data(temporary_path)
        report = run_audit(market_data, config)
    except ValueError as error:
        st.error(str(error))
        return

    st.session_state["audit_report"] = report
    render_report(report)


if __name__ == "__main__":
    run_app()
