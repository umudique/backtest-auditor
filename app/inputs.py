"""Input handling contracts for the Streamlit UI boundary."""

from typing import Any

from src.contracts import AuditConfig, MarketData
from src.data.canonical import Canonicalizer
from src.data.loader import MarketDataLoader
from src.data.schema import SchemaValidator
from src.data.validation import TimeSeriesValidator, ValueValidator


def validate_upload(file_path: str) -> str:
    """Validate a user-supplied upload path at the UI boundary.

    Args:
        file_path: Path selected by the user for market-data upload. Only CSV
            and Parquet paths are accepted by the MVP UI.

    Returns:
        The validated path unchanged.

    Raises:
        NotImplementedError: Until Phase 2 supplies surface-level validation.
        ValueError: If the path does not end with ``.csv`` or ``.parquet``.

    Invariants:
        This function performs UX-level file-extension validation only.
        It does not read files, parse data, validate OHLCV structure, call the
        data layer, or perform analytical validation.
    """
    lowered_path = file_path.lower()
    if lowered_path.endswith((".csv", ".parquet")):
        return file_path

    extension = file_path.rsplit(".", maxsplit=1)[-1] if "." in file_path else ""
    raise ValueError(f"unsupported upload extension: {extension}")


def load_market_data(file_path: str) -> MarketData:
    """Load and canonicalize uploaded market data through the data layer.

    Args:
        file_path: Local path to a user-uploaded CSV or Parquet file.

    Returns:
        Canonical ``MarketData`` produced by the data-layer canonicalizer.

    Raises:
        ValueError: If upload validation, schema validation, time-series
            validation, value validation, or canonicalization rejects the input.

    Invariants:
        Calls ``validate_upload`` before file loading.
        Performs no financial calculations, signal generation, cost handling,
        metric calculations, validation analysis, or reporting logic.
        ``MarketData`` is constructed only by ``Canonicalizer``.
    """
    validated_path = validate_upload(file_path)
    data = MarketDataLoader().load(validated_path)
    data = SchemaValidator().validate(data)
    data = TimeSeriesValidator().validate(data)
    data = ValueValidator().validate(data)
    return Canonicalizer().canonicalize(data)


def build_config(form_values: dict[str, Any]) -> AuditConfig:
    """Build an ``AuditConfig`` from UI form values.

    Args:
        form_values: Raw form mapping collected by the UI. Required values
            include strategy, cost, slippage, position sizing, execution,
            validation, and random-seed inputs.

    Returns:
        A canonical ``AuditConfig`` from ``src.contracts``.

    Raises:
        NotImplementedError: Until Phase 2 supplies form mapping.
        ValueError: If required fields are missing, if ``random_seed`` is
            missing or non-integer, or if form data cannot be mapped to the
            canonical configuration contract.

    Invariants:
        This function does not redefine ``AuditConfig``.
        Validation is limited to surface-level required-field and type checks.
        Analytical validation remains in the data, engine, validation, and
        metrics layers.
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
