"""Canonical market data construction contracts."""

import pandas as pd

from src.contracts import MarketData


class Canonicalizer:
    """Convert fully validated OHLCV data into canonical ``MarketData``.

    Invariants:
        This is the only data-layer component that constructs ``MarketData``.
        Output fields follow ``MarketData`` field order: ``timestamp``,
        ``open``, ``high``, ``low``, ``close``, ``volume``.
        Output is sorted by a DatetimeIndex in ascending order.
        Canonicalization does not calculate signals, costs, returns, validation
        metrics, risk metrics, or reporting conclusions.
    """

    def canonicalize(self, data: pd.DataFrame) -> MarketData:
        """Construct canonical ``MarketData`` from fully validated OHLCV rows.

        Args:
            data: DataFrame accepted by ``SchemaValidator``,
                ``TimeSeriesValidator``, and ``ValueValidator``.

        Returns:
            A ``MarketData`` instance whose series are aligned to the same
            ascending ``DatetimeIndex`` and ordered according to the canonical
            field contract.

        Raises:
            ValueError: If required columns are absent, timestamps cannot form a
                DatetimeIndex, rows cannot be sorted into a valid chronological
                canonical representation, or any canonical field would be
                missing from the output.

        Invariants:
            Canonicalization is the single construction point for ``MarketData``
            inside ``src/data``.
            The returned index is a pandas ``DatetimeIndex`` sorted ascending.
            The returned data contains no extra signal, cost, metric, validation,
            or reporting fields.
            Failure emits no partial ``MarketData``.
        """
        required_columns = ("timestamp", "open", "high", "low", "close", "volume")
        missing_columns = [column for column in required_columns if column not in data.columns]
        if missing_columns:
            raise ValueError(f"missing required columns: {', '.join(missing_columns)}")

        canonical = data[list(required_columns)].copy()
        try:
            canonical["timestamp"] = pd.to_datetime(canonical["timestamp"])
        except (TypeError, ValueError) as exc:
            raise ValueError("timestamp cannot form a DatetimeIndex") from exc

        canonical = canonical.sort_values(by="timestamp")
        index = pd.DatetimeIndex(canonical["timestamp"])

        return MarketData(
            timestamp=pd.Series(index, index=index, name="timestamp"),
            open=pd.Series(canonical["open"].to_numpy(), index=index, name="open"),
            high=pd.Series(canonical["high"].to_numpy(), index=index, name="high"),
            low=pd.Series(canonical["low"].to_numpy(), index=index, name="low"),
            close=pd.Series(canonical["close"].to_numpy(), index=index, name="close"),
            volume=pd.Series(canonical["volume"].to_numpy(), index=index, name="volume"),
        )
