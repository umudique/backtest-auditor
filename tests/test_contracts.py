"""Phase 1 tests for canonical domain contracts."""

from __future__ import annotations

import dataclasses
import inspect
from typing import get_type_hints

import pytest

from src.contracts import (
    AuditConfig,
    AuditReport,
    BacktestResult,
    MarketData,
    ValidationResult,
)

PUBLIC_CONTRACTS = (
    AuditConfig,
    MarketData,
    BacktestResult,
    ValidationResult,
    AuditReport,
)


def _field_names(contract_type: type[object]) -> list[str]:
    return [field.name for field in dataclasses.fields(contract_type)]


@pytest.mark.parametrize("contract_type", PUBLIC_CONTRACTS)
def test_public_contracts_are_dataclasses(contract_type: type[object]) -> None:
    assert dataclasses.is_dataclass(contract_type)


@pytest.mark.parametrize("contract_type", PUBLIC_CONTRACTS)
def test_public_contracts_have_type_hints_for_every_field(contract_type: type[object]) -> None:
    hints = get_type_hints(contract_type)

    assert set(hints) == set(_field_names(contract_type))


@pytest.mark.parametrize("contract_type", PUBLIC_CONTRACTS)
def test_public_contract_docstrings_record_preconditions_and_invariants(
    contract_type: type[object],
) -> None:
    docstring = inspect.getdoc(contract_type)

    assert docstring is not None
    assert "Args:" in docstring
    assert "Raises:" in docstring
    assert "Invariants:" in docstring


def test_audit_config_fields_match_architecture_section_8_1() -> None:
    assert _field_names(AuditConfig) == [
        "strategy_name",
        "strategy_parameters",
        "fee_assumption",
        "slippage_assumption",
        "position_sizing",
        "execution_assumptions",
        "validation_configuration",
        "random_seed",
    ]
    assert get_type_hints(AuditConfig)["random_seed"] is int


def test_market_data_fields_match_architecture_section_8_2() -> None:
    assert _field_names(MarketData) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]


def test_backtest_result_fields_match_architecture_section_8_3() -> None:
    assert _field_names(BacktestResult) == [
        "positions",
        "trades",
        "gross_returns",
        "net_returns",
        "equity_curve",
        "drawdown_series",
        "execution_metadata",
    ]


def test_backtest_result_rejects_same_gross_and_net_returns_object() -> None:
    returns: list[float] = [0.01, -0.02, 0.03]

    with pytest.raises(ValueError, match="gross_returns.*net_returns"):
        BacktestResult(
            positions=[0, 1, 1],
            trades=[],
            gross_returns=returns,
            net_returns=returns,
            equity_curve=[1.0, 0.98, 1.01],
            drawdown_series=[0.0, -0.02, 0.0],
            execution_metadata={"execution_timing": "signal_close_execute_next_open"},
        )


def test_validation_result_fields_match_architecture_section_8_4() -> None:
    assert _field_names(ValidationResult) == [
        "in_sample_metrics",
        "out_of_sample_metrics",
        "walk_forward_results",
        "sensitivity_results",
        "bootstrap_results",
        "monte_carlo_results",
    ]


def test_validation_result_requires_distinct_in_sample_and_oos_fields() -> None:
    fields = _field_names(ValidationResult)

    assert "in_sample_metrics" in fields
    assert "out_of_sample_metrics" in fields
    assert "sample_metrics" not in fields
    assert "combined_metrics" not in fields
    assert "merged_metrics" not in fields


def test_audit_report_fields_match_architecture_section_8_5() -> None:
    assert _field_names(AuditReport) == [
        "baseline_metrics",
        "cost_adjusted_metrics",
        "out_of_sample_metrics",
        "walk_forward_summary",
        "parameter_sensitivity_summary",
        "simulated_drawdown_summary",
        "regime_summary",
        "fragility_summary",
        "verdict",
        "charts",
    ]


def test_audit_report_verdict_has_no_default() -> None:
    verdict_field = next(
        field for field in dataclasses.fields(AuditReport) if field.name == "verdict"
    )

    assert verdict_field.default is dataclasses.MISSING
    assert verdict_field.default_factory is dataclasses.MISSING
