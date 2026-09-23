"""Local demo entry point — not committed. Adds sample-data shortcut."""

from __future__ import annotations

import pathlib
import tempfile

import streamlit as st

from app.display import render_report
from app.inputs import build_config, load_market_data

_SAMPLE_CSV = pathlib.Path(__file__).parent / "examples" / "spy_daily.csv"


def main() -> None:
    st.set_page_config(page_title="Backtest Auditor", layout="wide")
    st.title("Backtest Auditor")

    col_upload, col_sample = st.columns([3, 1])
    with col_upload:
        uploaded = st.file_uploader("Upload market data", type=["csv", "parquet"])
    with col_sample:
        st.write("")
        st.write("")
        if st.button("Load sample (SPY 2022–2024)"):
            st.session_state["demo_path"] = str(_SAMPLE_CSV)
            st.rerun()

    demo_path: str | None = st.session_state.get("demo_path")
    active_path: str | None = None
    if uploaded is not None:
        suffix = pathlib.Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            active_path = tmp.name
    elif demo_path is not None:
        active_path = demo_path

    if active_path:
        label = uploaded.name if uploaded else _SAMPLE_CSV.name
        st.info(f"Market data loaded: {label}")

    with st.form("audit_configuration"):
        strategy_name = st.text_input("Strategy name", value="moving_average_crossover")
        col1, col2 = st.columns(2)
        with col1:
            short_window = st.number_input("Short window", min_value=1, value=10, step=1)
            fee_rate = st.number_input("Fee rate", min_value=0.0, value=0.001, format="%.6f")
            random_seed = st.number_input("Random seed", min_value=0, value=42, step=1)
        with col2:
            long_window = st.number_input("Long window", min_value=2, value=50, step=1)
            slippage_rate = st.number_input(
                "Slippage rate", min_value=0.0, value=0.001, format="%.6f"
            )
        submitted = st.form_submit_button("Run audit")

    if not submitted:
        return

    if active_path is None:
        st.error("Upload market data or load sample data before running an audit.")
        return

    try:
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
        market_data = load_market_data(active_path)

        from src.orchestrator import run_audit

        report = run_audit(market_data, config)

    except ValueError as error:
        st.error(str(error))
        return

    st.session_state["audit_report"] = report
    render_report(report)


if __name__ == "__main__":
    main()
