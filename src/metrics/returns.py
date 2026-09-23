"""Return and expectancy metric contracts."""

import math

from src.contracts import BacktestResult


class ReturnMetrics:
    """Calculate return-based metrics from canonical backtest results.

    Invariants:
        Metrics consume ``BacktestResult.net_returns`` and never rerun the
        engine, strategy, execution, cost model, validation, reporting, or UI.
        Annualization factors are explicit parameters, never hidden constants.
        Sharpe and Sortino denominators never silently default to 1 when zero.
    """

    def sharpe_ratio(
        self,
        backtest_result: BacktestResult,
        risk_free_rate: float,
        annualization_factor: float,
    ) -> float:
        """Calculate annualized Sharpe ratio from net returns.

        Args:
            backtest_result: Canonical engine output containing ``net_returns``.
            risk_free_rate: Per-period risk-free rate subtracted from net
                returns before calculating mean excess return.
            annualization_factor: Explicit annualization factor used to scale
                the ratio.

        Returns:
            Annualized Sharpe ratio.

        Raises:
            ValueError: If returns are empty, ``annualization_factor`` is not
                positive, or the excess-return standard deviation is zero.

        Invariants:
            Uses ``BacktestResult.net_returns`` only.
            Does not apply costs, recalculate backtests, or inspect validation
            internals.
            The denominator must not silently default to 1.
        """
        returns = backtest_result.net_returns.astype("float64")
        if returns.empty:
            raise ValueError("returns must not be empty")
        if annualization_factor <= 0:
            raise ValueError("annualization_factor must be positive")

        excess_returns = returns - risk_free_rate
        denominator = float(excess_returns.std(ddof=1))
        if denominator == 0 or math.isnan(denominator):
            raise ValueError("standard deviation denominator is zero")

        return float(excess_returns.mean() / denominator * math.sqrt(annualization_factor))

    def sortino_ratio(
        self,
        backtest_result: BacktestResult,
        target_return: float,
        annualization_factor: float,
    ) -> float:
        """Calculate annualized Sortino ratio from net returns.

        Args:
            backtest_result: Canonical engine output containing ``net_returns``.
            target_return: Per-period target return used to calculate downside
                deviations.
            annualization_factor: Explicit annualization factor used to scale
                the ratio.

        Returns:
            Annualized Sortino ratio.

        Raises:
            ValueError: If returns are empty, ``annualization_factor`` is not
                positive, or downside deviation is zero.

        Invariants:
            Uses downside deviation only; it must not silently fall back to
            Sharpe or default denominator 1 when there are no negative returns.
            Does not rerun the engine or calculate reporting output.
        """
        returns = backtest_result.net_returns.astype("float64")
        if returns.empty:
            raise ValueError("returns must not be empty")
        if annualization_factor <= 0:
            raise ValueError("annualization_factor must be positive")

        downside_returns = returns[returns < target_return]
        denominator = float(downside_returns.std(ddof=1))
        if denominator == 0 or math.isnan(denominator):
            raise ValueError("downside denominator is zero")

        return float(returns.mean() / denominator * math.sqrt(annualization_factor))

    def expectancy(self, backtest_result: BacktestResult) -> float:
        """Calculate trade/return expectancy from net returns.

        Args:
            backtest_result: Canonical engine output containing ``net_returns``.

        Returns:
            Expectancy as ``win_rate * average_win - loss_rate * average_loss``.

        Raises:
            ValueError: If net returns are empty or wins/losses cannot be
                separated into a meaningful expectancy calculation.

        Invariants:
            Operates on existing net returns only.
            Does not recalculate trades, costs, validation, or reporting.
        """
        returns = backtest_result.net_returns.astype("float64")
        if returns.empty:
            raise ValueError("returns must not be empty")

        wins = returns[returns > 0]
        losses = returns[returns < 0]
        win_rate = len(wins) / len(returns)
        loss_rate = len(losses) / len(returns)
        average_win = float(wins.mean()) if not wins.empty else 0.0
        average_loss = abs(float(losses.mean())) if not losses.empty else 0.0
        return float(win_rate * average_win - loss_rate * average_loss)
