"""Market data file loading contracts.

The data loader boundary is limited to reading supported file formats into a
tabular representation. It does not validate schema, normalize ordering,
construct ``MarketData``, calculate signals, apply costs, or calculate metrics.
"""

from pathlib import Path

import pandas as pd


class MarketDataLoader:
    """Read raw OHLCV files from disk.

    Invariants:
        Supports CSV and Parquet inputs only.
        Rejects unsupported file extensions before attempting to read content.
        Returns raw tabular data only; it never constructs ``MarketData`` and
        never performs schema, time-series, value, signal, cost, or metric work.
    """

    def load(self, path: str | Path) -> pd.DataFrame:
        """Load a CSV or Parquet OHLCV file as a raw DataFrame.

        Args:
            path: Filesystem path to a local ``.csv`` or ``.parquet`` market
                data file.

        Returns:
            A pandas DataFrame containing the raw columns exactly as read from
            the supported input file.

        Raises:
            ValueError: If ``path`` has any extension other than ``.csv`` or
                ``.parquet``. The error must be specific enough to identify the
                unsupported extension and must be raised before any read is
                attempted.
            FileNotFoundError: If a supported path does not exist.
            OSError: If pandas cannot read an otherwise supported local file.

        Invariants:
            Loading has no side effects beyond reading the requested file.
            Loading does not validate required OHLCV fields.
            Loading does not sort, coerce, or canonicalize data.
            Loading does not emit partial canonical output on failure.
        """
        file_path = Path(path)
        suffix = file_path.suffix.lower()

        if suffix not in {".csv", ".parquet"}:
            raise ValueError("unsupported file extension; expected CSV or Parquet")

        if suffix == ".csv":
            return pd.read_csv(file_path)

        return pd.read_parquet(file_path)
