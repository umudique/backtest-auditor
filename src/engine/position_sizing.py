"""Position sizing contracts for the backtest engine."""

from typing import Any

import pandas as pd


class PositionSizer:
    """Convert strategy signals into bounded positions.

    Invariants:
        Position sizing consumes strategy output and explicit sizing
        configuration only.
        Position sizing does not apply transaction costs, execution fills,
        portfolio accounting, metrics, or reporting.
        Output positions remain aligned with the input signal series.
    """

    def size_positions(self, signals: pd.Series, config: dict[str, Any]) -> pd.Series:
        """Convert signal values into bounded target positions.

        Args:
            signals: Strategy signal series aligned with canonical market data.
            config: Explicit sizing configuration, including configured lower
                and upper exposure bounds.

        Returns:
            Position series with the same index and length as ``signals``.

        Raises:
            ValueError: If sizing bounds are missing, invalid, or if generated
                positions would exceed configured bounds.

        Invariants:
            Output length equals input signal length.
            Every position is within configured bounds.
            This method does not inspect future market bars and does not apply
            fees, slippage, execution timing, metrics, or reporting logic.
        """
        min_position = config.get("min_position")
        max_position = config.get("max_position")

        if not isinstance(min_position, int | float) or not isinstance(max_position, int | float):
            raise ValueError("min_position and max_position must be configured")
        if min_position > max_position:
            raise ValueError("min_position must not exceed max_position")

        return signals.clip(lower=float(min_position), upper=float(max_position))
