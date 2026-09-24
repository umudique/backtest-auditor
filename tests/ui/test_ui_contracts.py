"""Phase 1 UI contract tests."""

from __future__ import annotations

import inspect

import pytest

from app.inputs import build_config, load_market_data, validate_upload
from src.contracts import AuditConfig


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


def test_validate_upload_rejects_txt_file() -> None:
    with pytest.raises(ValueError):
        validate_upload("prices.txt")


def test_load_market_data_raises_for_invalid_extension() -> None:
    with pytest.raises(ValueError):
        load_market_data("prices.txt")


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


def test_render_report_does_not_import_analytical_layers() -> None:
    import app.display as display_module

    source = inspect.getsource(display_module)

    assert "src.engine" not in source
    assert "src.validation" not in source
    assert "src.metrics" not in source


def test_run_app_does_not_import_analytical_layers() -> None:
    import app.main as main_module

    source = inspect.getsource(main_module)

    assert "src.engine" not in source
    assert "src.validation" not in source
    assert "src.metrics" not in source


def test_main_calls_orchestrator_run_audit_not_pipeline_stages_directly() -> None:
    import app.main as main_module

    source = inspect.getsource(main_module)

    assert "from src.orchestrator import run_audit" in source
    assert "run_audit" in source
    assert "src.engine" not in source
    assert "src.validation" not in source
    assert "src.metrics" not in source
