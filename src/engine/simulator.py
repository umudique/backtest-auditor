"""Portfolio simulation contracts for the backtest engine."""

from typing import Any

import pandas as pd

from src.contracts import MarketData


class PortfolioSimulator:
    """Produce positions, trades, returns, equity, and drawdown series.

    Invariants:
        Simulation consumes canonical market data, executed positions, and
        explicit configuration.
        Simulation advances chronologically and must not access bars beyond the
        current simulation step.
        Simulation does not apply fees or slippage; costs belong to
        ``CostModel``.
        Simulation does not calculate external metrics or reporting output.
    """

    def simulate(
        self,
        market_data: MarketData,
        executed_positions: pd.Series,
        config: dict[str, Any],
    ) -> dict[str, pd.Series | pd.DataFrame]:
        """Simulate gross portfolio path before transaction costs.

        Args:
            market_data: Canonical OHLCV input.
            executed_positions: Position series after execution timing is
                applied.
            config: Explicit simulator assumptions such as initial equity and
                trade-record conventions.

        Returns:
            Mapping containing ``positions``, ``trades``, ``gross_returns``,
            ``equity_curve``, and ``drawdown_series``.

        Raises:
            ValueError: If positions are not aligned to market data, if required
                simulator assumptions are missing, or if the simulation would
                require future bars.

        Invariants:
            ``equity_curve`` starts at ``1.0`` for normalized MVP output unless
            an explicit initial-equity convention says otherwise.
            ``drawdown_series`` is less than or equal to zero throughout.
            Position output length equals market-data length.
            No fees, slippage, metrics, reporting, or validation logic is
            performed here.
        """
        positions = executed_positions.reindex(market_data.close.index)
        if positions.isna().any():
            raise ValueError("executed positions must align to market data")

        initial_equity = float(config.get("initial_equity", 1.0))
        close_returns = market_data.close.pct_change().fillna(0.0)
        gross_returns = positions.astype(float) * close_returns

        equity_curve = (1.0 + gross_returns).cumprod() * initial_equity
        if not equity_curve.empty:
            equity_curve.iloc[0] = initial_equity

        running_peak = equity_curve.cummax()
        drawdown_series = (equity_curve / running_peak) - 1.0
        drawdown_series = drawdown_series.fillna(0.0).clip(upper=0.0)

        trades = pd.DataFrame(
            {"trade_size": positions.astype(float).diff().fillna(positions.astype(float)).abs()},
            index=market_data.close.index,
        )

        return {
            "positions": positions,
            "trades": trades,
            "gross_returns": gross_returns,
            "equity_curve": equity_curve,
            "drawdown_series": drawdown_series,
        }
