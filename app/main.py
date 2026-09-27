"""Top-level Streamlit application contract."""

import pathlib
import tempfile

import streamlit as st

from app.display import render_report
from app.inputs import build_config, load_and_run_audit

_CSS = """
<style>
@font-face {
    font-family: 'IBM Plex Sans';
    font-weight: 400;
    src: url('/app/static/fonts/IBMPlexSans-Regular.woff2') format('woff2');
}
@font-face {
    font-family: 'IBM Plex Sans';
    font-weight: 500;
    src: url('/app/static/fonts/IBMPlexSans-Medium.woff2') format('woff2');
}
@font-face {
    font-family: 'IBM Plex Sans';
    font-weight: 600;
    src: url('/app/static/fonts/IBMPlexSans-SemiBold.woff2') format('woff2');
}
@font-face {
    font-family: 'IBM Plex Sans';
    font-weight: 700;
    src: url('/app/static/fonts/IBMPlexSans-Bold.woff2') format('woff2');
}
@font-face {
    font-family: 'IBM Plex Mono';
    font-weight: 400;
    src: url('/app/static/fonts/IBMPlexMono-Regular.woff2') format('woff2');
}
@font-face {
    font-family: 'IBM Plex Mono';
    font-weight: 600;
    src: url('/app/static/fonts/IBMPlexMono-SemiBold.woff2') format('woff2');
}

/* Base: body */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 400;
}
/* Headings */
h1, h2, h3, h4,
[data-testid="stHeading"],
.stSubheader, .stTitle {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
}
/* Metrics: value + label */
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
[data-testid="stMetricDelta"],
[data-testid="stCaption"],
code, pre, .stCodeBlock {
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stMetricValue"],
.metric-value {
    font-size: 1.55rem !important;
    font-weight: 400 !important;
}
[data-testid="stMetricLabel"],
.metric-label {
    font-size: 0.85rem !important;
    font-weight: 400 !important;
}
.metric-value.bold { font-weight: 700 !important; }
/* Table cells */
table td, table th {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
}
/* Small secondary button (← New Audit) */
button[kind="secondary"] {
    padding: 0.1rem 0.5rem !important;
    font-size: 0.72rem !important;
    min-height: unset !important;
    line-height: 1.4 !important;
}
/* Tighten heading → caption gap */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] strong {
    margin-bottom: 0 !important;
}
[data-testid="stCaptionContainer"],
[data-testid="stCaption"] {
    margin-top: 0.1rem !important;
    margin-bottom: 0.35rem !important;
}
#MainMenu, footer, header, .stDeployButton { visibility: hidden; }
.block-container {
    padding-top: 0.75rem !important;
    padding-bottom: 0 !important;
    max-width: 100% !important;
}
[data-testid="column"]:nth-child(2) {
    overflow-y: auto;
    max-height: 62vh;
}
</style>
"""


def run_app() -> None:
    """Run the verdict-first Streamlit UI entry point.

    Args:
        None.

    Returns:
        None.

    Raises:
        ValueError: If user-provided inputs fail surface-level UI validation.
        FileNotFoundError: If an accepted upload path cannot be found after
            Streamlit writes it to a temporary file.

    Invariants:
        The UI is a thin presentation and orchestration boundary.
        It reaches orchestration through ``run_audit_from_file`` and renders
        the returned ``AuditReport`` without calling engine, validation,
        metrics, or data modules directly.
        Session state holds only ``AuditConfig`` and ``AuditReport`` objects.
        The rendered report follows the verdict-first layer order from
        ``docs/ui_architecture.md``.
    """
    st.markdown(_CSS, unsafe_allow_html=True)

    if st.session_state.get("audit_report") is not None:
        render_report(
            st.session_state["audit_report"],
            st.session_state.get("audit_config"),
        )
        return

    _render_input_form()


def _render_input_form() -> None:
    """Render the landing input form in the main area."""
    st.title("Backtest Auditor")
    st.caption("Upload historical market data and configure a strategy to receive a trust verdict.")

    uploaded = st.file_uploader("Market data (CSV or Parquet)", type=["csv", "parquet"])

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        short_window = st.number_input("Short window", min_value=1, value=10, step=1)
        fee_rate = st.number_input("Fee rate", min_value=0.0, value=0.001, format="%.6f")
    with col_b:
        long_window = st.number_input("Long window", min_value=2, value=50, step=1)
        slippage_rate = st.number_input("Slippage rate", min_value=0.0, value=0.001, format="%.6f")
    with col_c:
        random_seed = st.number_input("Random seed", min_value=0, value=42, step=1)
        strategy_name = st.text_input("Strategy", value="moving_average_crossover")

    submitted = st.button("Run audit", type="primary", use_container_width=True)

    if not submitted:
        return

    if uploaded is None:
        st.error("Upload market data before running an audit.")
        return

    suffix = pathlib.Path(uploaded.name).suffix
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            temporary_path = tmp.name

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
        st.session_state["audit_config"] = config
        report = load_and_run_audit(temporary_path, config)
    except (FileNotFoundError, OSError, ValueError) as error:
        st.error(str(error))
        return

    st.session_state["audit_report"] = report
    st.rerun()


if __name__ == "__main__":
    run_app()
