"""Surface-level input contracts for the Streamlit UI boundary."""

from typing import Any

from src.contracts import AuditConfig, AuditReport
from src.orchestrator import run_audit_from_file


def validate_upload(file_path: str) -> str:
    """Validate a user-supplied upload path at the UI boundary.

    Args:
        file_path: Path selected by the user for market-data upload. The MVP UI
            accepts CSV and Parquet file names only.

    Returns:
        The validated path unchanged.

    Raises:
        ValueError: If the path does not end with ``.csv`` or ``.parquet``.

    Invariants:
        This is UX validation only.
        It does not read files, inspect OHLCV fields, canonicalize data, or
        import the data layer.
    """
    lowered_path = file_path.lower()
    if lowered_path.endswith((".csv", ".parquet")):
        return file_path

    extension = file_path.rsplit(".", maxsplit=1)[-1] if "." in file_path else ""
    raise ValueError(f"unsupported upload extension: {extension}")


def load_and_run_audit(file_path: str, config: AuditConfig) -> AuditReport:
    """Validate an upload path and delegate audit execution to orchestration.

    Args:
        file_path: Local path for a CSV or Parquet upload.
        config: Canonical audit configuration collected by the UI.

    Returns:
        Canonical ``AuditReport`` returned by the orchestrator.

    Raises:
        ValueError: If surface-level upload validation fails.
        FileNotFoundError: If the accepted path does not exist.
        OSError: If the accepted path cannot be read.

    Invariants:
        The UI does not import or call the analytical data package directly.
        Analytical validation remains outside ``app``.
    """
    validated_path = validate_upload(file_path)
    return run_audit_from_file(validated_path, config)


def load_market_data(file_path: str, config: AuditConfig | None = None) -> AuditReport:
    """Compatibility wrapper for the UI upload-to-audit boundary.

    Args:
        file_path: Local path for a CSV or Parquet upload.
        config: Canonical audit configuration collected by the UI.

    Returns:
        Canonical ``AuditReport`` returned by the orchestrator.

    Raises:
        ValueError: If surface-level upload validation fails or ``config`` is
            absent.

    Invariants:
        The UI does not import the analytical data package directly.
        The orchestrator owns data loading and canonicalization.
    """
    validate_upload(file_path)
    if config is None:
        raise ValueError("AuditConfig is required to run an audit from a file")
    return load_and_run_audit(file_path, config)


def build_config(form_values: dict[str, Any]) -> AuditConfig:
    """Build an ``AuditConfig`` from UI form values.

    Args:
        form_values: Raw form mapping collected by the UI. Required values
            include strategy, cost, slippage, position sizing, execution,
            validation, and random-seed inputs.

    Returns:
        A canonical ``AuditConfig`` from ``src.contracts``.

    Raises:
        ValueError: If required fields are missing, if ``random_seed`` is
            missing or non-integer, or if form data cannot be mapped to the
            canonical configuration contract.

    Invariants:
        This function does not redefine ``AuditConfig``.
        Validation is limited to surface-level required-field and type checks.
        It performs no financial calculations and no analytical validation.
    """
    if "strategy_name" not in form_values:
        raise ValueError("strategy_name is required")

    random_seed = form_values.get("random_seed")
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise ValueError("random_seed must be an integer")

    return AuditConfig(
        strategy_name=str(form_values["strategy_name"]),
        strategy_parameters=dict(form_values.get("strategy_parameters", {})),
        fee_assumption=form_values.get("fee_assumption", {}),
        slippage_assumption=form_values.get("slippage_assumption", {}),
        position_sizing=form_values.get(
            "position_sizing",
            {"min_position": -1.0, "max_position": 1.0},
        ),
        execution_assumptions=dict(form_values.get("execution_assumptions", {})),
        validation_configuration=dict(form_values.get("validation_configuration", {})),
        random_seed=random_seed,
    )
