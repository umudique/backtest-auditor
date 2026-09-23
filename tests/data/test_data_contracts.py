"""Phase 1 tests for data loading, validation, and canonicalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.contracts import MarketData
from src.data.canonical import Canonicalizer
from src.data.loader import MarketDataLoader
from src.data.schema import SchemaValidator
from src.data.validation import TimeSeriesValidator, ValueValidator

CANONICAL_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def valid_ohlcv_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-02 09:30", "2024-01-02 09:31", "2024-01-02 09:32"]
            ),
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.5, 103.0],
            "low": [99.5, 100.5, 101.5],
            "close": [100.5, 102.0, 102.5],
            "volume": [1000, 1100, 1200],
        },
        columns=CANONICAL_COLUMNS,
    )


def write_csv_fixture(tmp_path: Path, data: pd.DataFrame) -> Path:
    path = tmp_path / "ohlcv.csv"
    data.to_csv(path, index=False)
    return path


def write_parquet_fixture(tmp_path: Path, data: pd.DataFrame) -> Path:
    path = tmp_path / "ohlcv.parquet"
    data.to_parquet(path, index=False)
    return path


def run_validation_pipeline(data: pd.DataFrame, canonicalizer: Canonicalizer) -> MarketData:
    schema_valid = SchemaValidator().validate(data)
    time_valid = TimeSeriesValidator().validate(schema_valid)
    value_valid = ValueValidator().validate(time_valid)
    return canonicalizer.canonicalize(value_valid)


def test_market_data_loader_reads_valid_csv(tmp_path: Path) -> None:
    source = valid_ohlcv_frame()
    path = write_csv_fixture(tmp_path, source)

    loaded = MarketDataLoader().load(path)

    assert loaded.shape == (3, 6)
    assert list(loaded.columns) == CANONICAL_COLUMNS


def test_market_data_loader_reads_valid_parquet(tmp_path: Path) -> None:
    source = valid_ohlcv_frame()
    path = write_parquet_fixture(tmp_path, source)

    loaded = MarketDataLoader().load(path)

    assert loaded.shape == (3, 6)
    assert list(loaded.columns) == CANONICAL_COLUMNS


def test_market_data_loader_rejects_unsupported_extension_before_reading(tmp_path: Path) -> None:
    path = tmp_path / "ohlcv.json"
    path.write_text("not read by the loader", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported.*extension|CSV.*Parquet"):
        MarketDataLoader().load(path)


def test_schema_validator_rejects_missing_required_ohlcv_field() -> None:
    data = valid_ohlcv_frame().drop(columns=["high"])

    with pytest.raises(ValueError, match="missing.*high|required.*high"):
        SchemaValidator().validate(data)


def test_schema_validator_rejects_wrong_column_types() -> None:
    data = valid_ohlcv_frame()
    data["close"] = ["bad", "price", "data"]

    with pytest.raises(TypeError, match="close|numeric"):
        SchemaValidator().validate(data)


def test_time_series_validator_rejects_duplicate_timestamps() -> None:
    data = valid_ohlcv_frame()
    data.loc[1, "timestamp"] = data.loc[0, "timestamp"]

    with pytest.raises(ValueError, match="duplicate.*timestamp"):
        TimeSeriesValidator().validate(data)


def test_time_series_validator_rejects_unordered_timestamps() -> None:
    data = valid_ohlcv_frame()
    data = data.iloc[[1, 0, 2]].reset_index(drop=True)

    with pytest.raises(ValueError, match="unordered|chronological|ascending"):
        TimeSeriesValidator().validate(data)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda data: data.__setitem__("high", [101.0, 99.0, 103.0]), "high.*low"),
        (lambda data: data.__setitem__("low", [99.5, 103.0, 101.5]), "low.*open|low.*close"),
        (lambda data: data.__setitem__("open", [100.0, -1.0, 102.0]), "positive.*price"),
        (lambda data: data.__setitem__("volume", [1000, 0, 1200]), "positive.*volume"),
    ],
)
def test_value_validator_rejects_invalid_ohlcv_structure(
    mutator: Any,
    message: str,
) -> None:
    data = valid_ohlcv_frame()
    mutator(data)

    with pytest.raises(ValueError, match=message):
        ValueValidator().validate(data)


def test_canonicalizer_sorts_output_and_uses_canonical_field_order() -> None:
    data = valid_ohlcv_frame().iloc[[2, 0, 1]].reset_index(drop=True)

    market_data = Canonicalizer().canonicalize(data)

    assert isinstance(market_data, MarketData)
    assert isinstance(market_data.close.index, pd.DatetimeIndex)
    assert market_data.close.index.is_monotonic_increasing
    assert market_data.timestamp.iloc[0] < market_data.timestamp.iloc[-1]
    assert market_data.timestamp.iloc[0] == pd.Timestamp("2024-01-02 09:30")
    assert market_data.timestamp.iloc[0] != data.loc[0, "timestamp"]


def test_canonicalizer_output_is_market_data_instance() -> None:
    market_data = Canonicalizer().canonicalize(valid_ohlcv_frame())

    assert isinstance(market_data, MarketData)


def test_validation_pipeline_fails_closed_without_partial_market_data() -> None:
    class RecordingCanonicalizer(Canonicalizer):
        called = False

        def canonicalize(self, data: pd.DataFrame) -> MarketData:
            self.called = True
            return super().canonicalize(data)

    data = valid_ohlcv_frame().drop(columns=["close"])
    canonicalizer = RecordingCanonicalizer()

    with pytest.raises(ValueError, match="missing.*close|required.*close"):
        run_validation_pipeline(data, canonicalizer)

    assert canonicalizer.called is False
