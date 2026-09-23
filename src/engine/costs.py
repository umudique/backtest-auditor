"""Transaction cost contracts for the backtest engine."""

from typing import Any

import pandas as pd


class CostModel:
    """Apply explicit fees and slippage to gross returns.

    Invariants:
        This is the only engine component that applies fees and slippage.
        Zero-cost runs must be explicit: fee and slippage assumptions are both
        present and equal to zero.
        Gross and net returns remain distinguishable series.
    """

    def apply_costs(
        self,
        gross_returns: pd.Series,
        trades: pd.DataFrame,
        cost_config: dict[str, Any],
    ) -> pd.Series:
        """Apply configured transaction costs to gross returns.

        Args:
            gross_returns: Period return series before fees and slippage.
            trades: Trade records containing enough information to attribute
                cost impact to trade size and turnover.
            cost_config: Explicit fee and slippage assumptions, including
                ``fee_rate`` and ``slippage_rate``.

        Returns:
            Net return series after applying explicit engine-owned costs.

        Raises:
            ValueError: If fee or slippage assumptions are missing, negative, or
                otherwise ambiguous.

        Invariants:
            Net returns equal gross returns minus explicit fee and slippage
            impact.
            If both fee and slippage are explicitly zero, net returns may equal
            gross returns by value but must still be a distinct object.
            No reporting, metrics, execution timing, or signal generation logic
            is performed here.
        """
        if "fee_rate" not in cost_config or "slippage_rate" not in cost_config:
            raise ValueError("explicit fee_rate and slippage_rate are required")

        fee_rate = float(cost_config["fee_rate"])
        slippage_rate = float(cost_config["slippage_rate"])
        if fee_rate < 0 or slippage_rate < 0:
            raise ValueError("fee_rate and slippage_rate must be non-negative")

        if "trade_size" not in trades.columns:
            raise ValueError("trades must include trade_size")

        trade_size = trades["trade_size"].reindex(gross_returns.index, fill_value=0.0)
        trade_size = trade_size.astype("float64").abs()
        total_cost = trade_size * (fee_rate + slippage_rate)
        gross = gross_returns.astype("float64")
        net_returns = gross - total_cost
        return pd.Series(net_returns, index=gross_returns.index)
