"""Time-series and OHLCV value validation contracts."""

import pandas as pd


class TimeSeriesValidator:
    """Validate timestamp integrity for chronological backtesting.

    Invariants:
        Duplicate timestamps and out-of-order timestamps are distinct failure
        modes with distinguishable errors.
        Time-series validation does not construct ``MarketData``.
        Time-series validation does not calculate signals, costs, or metrics.
    """

    def validate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate timestamp uniqueness and chronological ordering.

        Args:
            data: Schema-valid DataFrame containing a ``timestamp`` column.

        Returns:
            The accepted DataFrame for value validation. Returned data remains
            tabular and non-canonical.

        Raises:
            ValueError: If any timestamp is duplicated. The error must identify
                duplicate timestamp rejection.
            ValueError: If timestamps are not strictly chronological ascending.
                The error must identify unordered timestamp rejection separately
                from duplicate timestamp rejection.

        Invariants:
            Invalid chronology fails before canonicalization.
            This validator rejects rather than repairs duplicate or unordered
            timestamps.
            No partial ``MarketData`` is emitted on failure.
        """
        timestamps = pd.to_datetime(data["timestamp"])

        if timestamps.duplicated().any():
            raise ValueError("duplicate timestamp values are not allowed")

        if not timestamps.is_monotonic_increasing:
            raise ValueError("timestamps must be chronological ascending")

        return data


class ValueValidator:
    """Validate structural OHLCV value integrity.

    Invariants:
        Structurally unusable observations are rejected before canonicalization.
        Value validation does not construct ``MarketData``.
        Value validation does not calculate signals, costs, returns, or metrics.
    """

    def validate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLCV relationships and positive value constraints.

        Args:
            data: Schema-valid and time-valid DataFrame with OHLCV columns.

        Returns:
            The accepted DataFrame for canonicalization.

        Raises:
            ValueError: If any row has ``high < low``.
            ValueError: If any row has ``low > open`` or ``low > close``.
            ValueError: If any price column contains a non-positive value.
            ValueError: If ``volume`` contains a zero or negative value.

        Invariants:
            Every accepted row is structurally usable for chronological
            backtesting.
            Validation fails closed: no partial canonical data is emitted when
            any OHLCV relationship or positivity check fails.
            The validator never fills, clips, repairs, or silently removes bad
            observations.
        """
        price_columns = ("open", "high", "low", "close")

        if (data.loc[:, price_columns] <= 0).any().any():
            raise ValueError("positive price values are required")

        if (data["volume"] <= 0).any():
            raise ValueError("positive volume values are required")

        if ((data["low"] > data["open"]) | (data["low"] > data["close"])).any():
            raise ValueError("low must not exceed open or close")

        if (data["high"] < data["low"]).any():
            raise ValueError("high must not be below low")

        return data
