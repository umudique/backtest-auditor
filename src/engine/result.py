"""Backtest result construction contracts for the engine."""

from typing import Any

import pandas as pd

from src.contracts import BacktestResult


class BacktestResultBuilder:
    """Build canonical ``BacktestResult`` objects from engine outputs.

    Invariants:
        This is the engine component that emits canonical ``BacktestResult``.
        Gross and net returns are always present and never the same object.
        Execution metadata is non-empty and records assumptions needed by
        reporting.
    """

    def build(
        self,
        positions: pd.Series,
        trades: pd.DataFrame,
        gross_returns: pd.Series,
        net_returns: pd.Series,
        equity_curve: pd.Series,
        drawdown_series: pd.Series,
        execution_metadata: dict[str, Any],
    ) -> BacktestResult:
        """Construct a canonical ``BacktestResult``.

        Args:
            positions: Final position series produced by the engine.
            trades: Normalized trade records.
            gross_returns: Period returns before fees and slippage.
            net_returns: Period returns after explicit ``CostModel`` fees and
                slippage.
            equity_curve: Normalized equity path.
            drawdown_series: Drawdown path aligned with equity.
            execution_metadata: Non-empty metadata recording strategy
                parameters, cost assumptions, sizing assumptions, execution
                timing, and other simulator assumptions.

        Returns:
            A canonical ``BacktestResult`` from ``src.contracts``.

        Raises:
            ValueError: If execution metadata is empty, if any required series
                is missing or misaligned, or if ``gross_returns is
                net_returns``.

        Invariants:
            The builder does not redefine ``BacktestResult``.
            The returned object preserves gross and net return transparency.
            All execution assumptions required by reporting are carried in
            ``execution_metadata``.
            No metrics, validation, reporting, or UI logic is performed here.
        """
        if not execution_metadata:
            raise ValueError("execution_metadata must be non-empty")
        if gross_returns is net_returns:
            raise ValueError("gross_returns and net_returns must not be the same object")

        return BacktestResult(
            positions=positions,
            trades=trades,
            gross_returns=gross_returns,
            net_returns=net_returns,
            equity_curve=equity_curve,
            drawdown_series=drawdown_series,
            execution_metadata=execution_metadata,
        )
