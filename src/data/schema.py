"""Schema validation contracts for raw OHLCV data."""

import pandas as pd


class SchemaValidator:
    """Validate required canonical OHLCV columns and their raw dtypes.

    Invariants:
        Required columns are exactly ``timestamp``, ``open``, ``high``, ``low``,
        ``close``, and ``volume``.
        Schema validation does not construct ``MarketData``.
        Schema validation does not sort rows, inspect duplicate timestamps,
        validate OHLCV relationships, calculate signals, apply costs, or
        calculate metrics.
    """

    required_columns: tuple[str, ...] = (
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    )

    def validate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate that raw data contains typed OHLCV columns.

        Args:
            data: Raw DataFrame from ``MarketDataLoader``.

        Returns:
            The accepted DataFrame for the next validation stage. The returned
            data must not be a partial ``MarketData`` object.

        Raises:
            ValueError: If one or more required OHLCV columns are absent. The
                error must identify missing columns.
            TypeError: If ``timestamp`` cannot be interpreted as temporal data
                or if any of ``open``, ``high``, ``low``, ``close``, or
                ``volume`` are not numeric. The error must identify the
                offending column.

        Invariants:
            All required columns are present before downstream validation runs.
            Wrong column types fail closed with a distinguishable type error.
            The validator never emits a canonical ``MarketData`` instance.
            The validator never drops, fills, or invents observations.
        """
        for column in self.required_columns:
            if column not in data.columns:
                raise ValueError(f"missing required column: {column}")

        try:
            pd.to_datetime(data["timestamp"])
        except (TypeError, ValueError) as exc:
            raise TypeError("timestamp must be temporal") from exc

        for column in ("open", "high", "low", "close", "volume"):
            if not pd.api.types.is_numeric_dtype(data[column]):
                raise TypeError(f"{column} must be numeric")

        return data
