"""Execution timing contracts for the backtest engine."""

from typing import Any

import pandas as pd


class ExecutionModel:
    """Apply explicit signal-to-fill timing assumptions.

    Invariants:
        Default timing is signal generated at bar T close and filled at bar T+1
        open.
        Any deviation from default timing must be explicit in configuration and
        later recorded in ``BacktestResult.execution_metadata``.
        Execution timing does not apply costs, calculate metrics, or report.
    """

    def apply_execution_timing(
        self,
        signals: pd.Series,
        open_prices: pd.Series,
        execution_assumptions: dict[str, Any],
    ) -> tuple[pd.Series, dict[str, Any]]:
        """Shift signals into executable positions under timing assumptions.

        Args:
            signals: Signal or target-position series generated at signal bars.
            open_prices: Open-price series aligned with canonical market data.
            execution_assumptions: Explicit timing assumptions from
                ``AuditConfig.execution_assumptions``.

        Returns:
            A tuple of ``(executed_positions, execution_metadata)``. Metadata
            records the timing assumption used, including the default
            ``signal_close_execute_next_open`` when no deviation is configured.

        Raises:
            ValueError: If execution assumptions are absent, ambiguous, or imply
                look-ahead fills.

        Invariants:
            Default execution uses signal at bar T close and fill at bar T+1
            open.
            Execution metadata is non-empty and sufficient for reporting to
            surface the timing assumption.
            This method does not apply fees, slippage, portfolio accounting,
            metrics, or reporting logic.
        """
        if not execution_assumptions or "timing" not in execution_assumptions:
            raise ValueError("execution assumptions must include timing")

        timing = execution_assumptions["timing"]
        if timing != "signal_close_execute_next_open":
            raise ValueError("unsupported execution timing assumption")

        executed_positions = signals.shift(1).reindex(open_prices.index).fillna(0)
        metadata = {"timing": timing}
        return executed_positions, metadata
