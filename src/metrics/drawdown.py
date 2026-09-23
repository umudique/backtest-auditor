"""Drawdown metric contracts."""

import pandas as pd

from src.contracts import BacktestResult


class DrawdownMetrics:
    """Calculate drawdown metrics from the equity curve.

    Invariants:
        Drawdown calculations derive from ``BacktestResult.equity_curve``, not
        from individual trade returns.
        Drawdown values are less than or equal to zero.
        Metrics do not rerun engine, validation, reporting, or UI logic.
    """

    def drawdown_series(self, backtest_result: BacktestResult) -> pd.Series:
        """Calculate drawdown series from a backtest equity curve.

        Args:
            backtest_result: Canonical engine output containing
                ``equity_curve``.

        Returns:
            Drawdown series aligned with ``backtest_result.equity_curve``.

        Raises:
            ValueError: If the equity curve is empty or starts at a
                non-positive value.

        Invariants:
            Every drawdown value is ``<= 0``.
            The calculation uses cumulative equity peaks, not trade returns.
        """
        equity_curve = backtest_result.equity_curve.astype("float64")
        if equity_curve.empty:
            raise ValueError("equity curve must not be empty")
        if float(equity_curve.iloc[0]) <= 0:
            raise ValueError("equity curve must start at a positive value")

        running_peak = equity_curve.cummax()
        drawdowns = (equity_curve / running_peak) - 1.0
        return drawdowns.clip(upper=0.0)

    def maximum_drawdown(self, backtest_result: BacktestResult) -> float:
        """Calculate maximum drawdown from the equity curve.

        Args:
            backtest_result: Canonical engine output containing
                ``equity_curve``.

        Returns:
            Minimum value of the derived drawdown series.

        Raises:
            ValueError: If the equity curve is empty or invalid.

        Invariants:
            Maximum drawdown is always ``<= 0``.
            The metric is derived from equity, not individual returns.
        """
        return float(self.drawdown_series(backtest_result).min())

    def average_drawdown(self, backtest_result: BacktestResult) -> float:
        """Calculate average drawdown from the equity curve.

        Args:
            backtest_result: Canonical engine output containing
                ``equity_curve``.

        Returns:
            Arithmetic average of the derived drawdown series.

        Raises:
            ValueError: If the equity curve is empty or invalid.

        Invariants:
            Average drawdown is based on the same equity-derived drawdown series
            used by ``maximum_drawdown``.
        """
        return float(self.drawdown_series(backtest_result).mean())
