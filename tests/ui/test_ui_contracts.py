"""Phase 1 verdict-first UI contract tests."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

import pytest

from app.inputs import build_config, validate_upload
from src.contracts import AuditConfig, AuditReport


@dataclass
class StreamlitCall:
    """Captured Streamlit call emitted by a fake presentation boundary."""

    name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclass
class FakeExpander:
    """Context manager returned by the fake Streamlit expander."""

    streamlit: FakeStreamlit
    label: str

    def __enter__(self) -> FakeStreamlit:
        self.streamlit.calls.append(StreamlitCall("enter_expander", (self.label,), {}))
        return self.streamlit

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.streamlit.calls.append(StreamlitCall("exit_expander", (self.label,), {}))


@dataclass
class FakeStreamlit:
    """Small Streamlit test double that records UI rendering calls."""

    calls: list[StreamlitCall] = field(default_factory=list)
    session_state: dict[str, object] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        def recorder(*args: Any, **kwargs: Any) -> object:
            self.calls.append(StreamlitCall(name, args, kwargs))
            if name == "expander":
                return FakeExpander(self, str(args[0]))
            return None

        return recorder


def _complete_form_values() -> dict[str, object]:
    return {
        "strategy_name": "moving_average_crossover",
        "strategy_parameters": {"short_window": 2, "long_window": 3},
        "fee_assumption": {"fee_rate": 0.001},
        "slippage_assumption": {"slippage_rate": 0.001},
        "position_sizing": {"min_position": -1.0, "max_position": 1.0},
        "execution_assumptions": {"timing": "signal_close_execute_next_open"},
        "validation_configuration": {"split_ratio": 0.6},
        "random_seed": 7,
    }


def _sample_report(
    *,
    sensitivity_grid_rows: list[dict[str, Any]] | None = None,
    walk_forward_summary: dict[str, Any] | None = None,
) -> AuditReport:
    return AuditReport(
        baseline_metrics={
            "gross_cumulative": 1.18,
            "max_drawdown": -0.184,
            "sharpe": 1.34,
            "net_return": 0.221,
        },
        cost_adjusted_metrics={
            "net_cumulative": 1.142,
            "net_return": 0.142,
            "sharpe": 0.82,
            "max_drawdown": -0.184,
        },
        out_of_sample_metrics={
            "sharpe": 0.82,
            "net_return": 0.093,
            "max_drawdown": -0.184,
        },
        walk_forward_summary=walk_forward_summary
        if walk_forward_summary is not None
        else {"folds": [{"fold": 1, "sharpe": 0.7}, {"fold": 2, "sharpe": 0.9}]},
        parameter_sensitivity_summary={},
        simulated_drawdown_summary={"median_max_drawdown": -0.227},
        regime_summary={},
        fragility_summary=["out-of-sample performance is weaker"],
        verdict="FRAGILE",
        charts=[
            {"type": "equity_curve", "series": ["Gross return", "Net return"]},
            {"type": "drawdown", "series": ["Net drawdown"]},
        ],
        sensitivity_grid_rows=sensitivity_grid_rows
        if sensitivity_grid_rows is not None
        else [{"short_window": 10, "long_window": 50, "sharpe": 0.82}],
    )


def _rendered_text(fake_streamlit: FakeStreamlit) -> str:
    return "\n".join(
        str(argument)
        for call in fake_streamlit.calls
        for argument in call.args
        if isinstance(argument, (str, int, float, dict, list, tuple))
    )


def test_validate_upload_rejects_txt_file() -> None:
    with pytest.raises(ValueError):
        validate_upload("prices.txt")


def test_validate_upload_rejects_xlsx_file() -> None:
    with pytest.raises(ValueError):
        validate_upload("prices.xlsx")


def test_validate_upload_accepts_csv_path() -> None:
    assert validate_upload("prices.csv") == "prices.csv"


def test_validate_upload_accepts_parquet_path() -> None:
    assert validate_upload("prices.parquet") == "prices.parquet"


def test_build_config_requires_strategy_name() -> None:
    form_values = _complete_form_values()
    del form_values["strategy_name"]

    with pytest.raises(ValueError, match="strategy_name"):
        build_config(form_values)


def test_build_config_requires_integer_random_seed() -> None:
    missing_seed = _complete_form_values()
    del missing_seed["random_seed"]
    with pytest.raises(ValueError, match="random_seed"):
        build_config(missing_seed)

    non_integer_seed = _complete_form_values()
    non_integer_seed["random_seed"] = "7"
    with pytest.raises(ValueError, match="random_seed"):
        build_config(non_integer_seed)


def test_build_config_returns_valid_audit_config() -> None:
    config = build_config(_complete_form_values())

    assert isinstance(config, AuditConfig)
    assert config.strategy_name == "moving_average_crossover"
    assert config.random_seed == 7


def test_verdict_banner_renders_report_verdict_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.display as display_module

    fake_streamlit = FakeStreamlit()
    monkeypatch.setattr(display_module, "st", fake_streamlit)

    display_module.render_report(_sample_report())

    rendered_text = _rendered_text(fake_streamlit)
    assert "BACKTEST VERDICT" in rendered_text
    assert "FRAGILE" in rendered_text
    assert "fragile" not in rendered_text


def test_kpi_strip_renders_all_five_required_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.display as display_module

    fake_streamlit = FakeStreamlit()
    monkeypatch.setattr(display_module, "st", fake_streamlit)

    display_module.render_report(_sample_report())

    rendered_text = _rendered_text(fake_streamlit)
    assert "Net Return" in rendered_text
    assert "OOS Sharpe" in rendered_text
    assert "Max Drawdown" in rendered_text
    assert "Cost Impact" in rendered_text
    assert "Monte Carlo DD" in rendered_text


def test_gross_and_net_equity_curves_are_present(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.display as display_module

    fake_streamlit = FakeStreamlit()
    monkeypatch.setattr(display_module, "st", fake_streamlit)

    display_module.render_report(_sample_report())

    rendered_text = _rendered_text(fake_streamlit)
    assert "Gross return" in rendered_text
    assert "Net return" in rendered_text


def test_is_oos_metric_table_includes_delta_column(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.display as display_module

    fake_streamlit = FakeStreamlit()
    monkeypatch.setattr(display_module, "st", fake_streamlit)

    display_module.render_report(_sample_report())

    rendered_text = _rendered_text(fake_streamlit)
    assert "In-sample" in rendered_text
    assert "Out-of-sample" in rendered_text
    assert "Delta" in rendered_text


def test_sensitivity_heatmap_renders_when_grid_rows_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.display as display_module

    fake_streamlit = FakeStreamlit()
    monkeypatch.setattr(display_module, "st", fake_streamlit)

    display_module.render_report(_sample_report())

    rendered_text = _rendered_text(fake_streamlit)
    assert "Sensitivity Heatmap" in rendered_text
    assert "short_window" in rendered_text
    assert "long_window" in rendered_text
    assert "sharpe" in rendered_text


def test_sensitivity_heatmap_placeholder_when_grid_rows_are_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.display as display_module

    fake_streamlit = FakeStreamlit()
    monkeypatch.setattr(display_module, "st", fake_streamlit)

    display_module.render_report(_sample_report(sensitivity_grid_rows=[]))

    rendered_text = _rendered_text(fake_streamlit)
    assert "Sensitivity Heatmap" in rendered_text
    assert "not yet computed" in rendered_text


def test_walk_forward_chart_renders_or_shows_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.display as display_module

    with_results = FakeStreamlit()
    monkeypatch.setattr(display_module, "st", with_results)
    display_module.render_report(_sample_report(walk_forward_summary={"folds": [{"sharpe": 0.7}]}))
    assert "Walk-forward" in _rendered_text(with_results)
    assert "sharpe" in _rendered_text(with_results)

    without_results = FakeStreamlit()
    monkeypatch.setattr(display_module, "st", without_results)
    display_module.render_report(_sample_report(walk_forward_summary={}))
    assert "Walk-forward" in _rendered_text(without_results)
    assert "not yet computed" in _rendered_text(without_results)


def test_ui_does_not_import_analytical_layers_directly() -> None:
    import app.display as display_module
    import app.inputs as inputs_module
    import app.main as main_module

    app_source = "\n".join(
        [
            inspect.getsource(display_module),
            inspect.getsource(inputs_module),
            inspect.getsource(main_module),
        ]
    )

    assert "src.engine" not in app_source
    assert "src.validation" not in app_source
    assert "src.metrics" not in app_source
    assert "src.data" not in app_source


def test_session_state_holds_only_audit_config_and_audit_report() -> None:
    import app.main as main_module

    source = inspect.getsource(main_module)

    assert 'session_state["audit_config"]' in source
    assert 'session_state["audit_report"]' in source
    assert "session_state[" not in source.replace(
        'session_state["audit_config"]',
        "",
    ).replace(
        'session_state["audit_report"]',
        "",
    )


def test_main_calls_orchestrator_not_pipeline_stages_directly() -> None:
    import app.main as main_module

    source = inspect.getsource(main_module)

    assert "load_and_run_audit" in source
    assert "src.engine" not in source
    assert "src.validation" not in source
    assert "src.metrics" not in source
    assert "src.data" not in source
