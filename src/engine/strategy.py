"""Strategy contracts for the backtest engine.

Strategies translate canonical market data and explicit strategy parameters
into signals or target positions. They do not apply costs, size positions,
simulate portfolios, calculate metrics, or render/report results.
"""

from typing import Any

import pandas as pd

from src.contracts import MarketData


class StrategyInterface:
    """Accepted strategy contract for the engine.

    Invariants:
        Strategies consume only canonical ``MarketData`` and explicit
        parameters.
        Strategies produce a signal or target-position series aligned to input
        data and never apply fees, slippage, reporting logic, or Streamlit UI.
        Strategy calculations must not use future bars for a signal at bar T.
    """

    def generate_signals(self, market_data: MarketData, parameters: dict[str, Any]) -> pd.Series:
        """Generate strategy signals from canonical market data.

        Args:
            market_data: Canonical OHLCV data from ``src.contracts.MarketData``.
            parameters: Explicit strategy parameters for this run.

        Returns:
            Signal or target-position series aligned one-to-one with
            ``market_data``. For the reference moving-average strategy,
            supported signal values are ``-1``, ``0``, and ``1``.

        Raises:
            NotImplementedError: When called on ``StrategyInterface`` directly.
            ValueError: If required parameters are absent, invalid, or imply
                look-ahead behavior.

        Invariants:
            Signal at bar T uses only market data available at or before bar T.
            Output length equals the input market-data length.
            No costs, sizing, execution fills, metrics, or reporting are
            calculated by this method.
        """
        raise NotImplementedError


class MovingAverageCrossoverStrategy(StrategyInterface):
    """Reference moving-average crossover strategy.

    Invariants:
        Exists to exercise the audit framework, not to discover or optimize a
        trading strategy.
        Produces deterministic ``-1``, ``0``, or ``1`` signals from close
        prices using only historical bars up to the signal timestamp.
        Does not apply fees, slippage, position sizing, reporting, or metrics.
    """

    def generate_signals(self, market_data: MarketData, parameters: dict[str, Any]) -> pd.Series:
        """Generate moving-average crossover signals.

        Args:
            market_data: Canonical OHLCV input whose ``close`` series is used
                for moving-average calculations.
            parameters: Explicit strategy parameters. Expected keys include
                ``short_window`` and ``long_window`` with positive integer
                lengths where ``short_window < long_window``.

        Returns:
            A pandas Series aligned to ``market_data.close``. Values are limited
            to ``-1``, ``0``, and ``1``.

        Raises:
            ValueError: If moving-average window parameters are missing,
                non-positive, non-integer, or ordered in a way that invalidates
                the crossover definition.

        Invariants:
            The signal at bar T is based only on close prices with index
            positions ``<= T``.
            The method does not shift signals into fills; execution timing
            belongs to ``ExecutionModel``.
            The method does not calculate costs, equity, drawdown, metrics, or
            report output.
        """
        short_window = parameters.get("short_window")
        long_window = parameters.get("long_window")

        if not isinstance(short_window, int) or not isinstance(long_window, int):
            raise ValueError("short_window and long_window must be integer parameters")
        if short_window <= 0 or long_window <= 0:
            raise ValueError("moving-average windows must be positive")
        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window")

        short_average = market_data.close.rolling(window=short_window).mean()
        long_average = market_data.close.rolling(window=long_window).mean()

        signals = pd.Series(0.0, index=market_data.close.index)
        signals = signals.mask(short_average > long_average, 1.0)
        signals = signals.mask(short_average < long_average, -1.0)
        signals = signals.mask(short_average.isna() | long_average.isna())
        return signals
