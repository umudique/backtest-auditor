"""Regime analysis metric contracts."""

import pandas as pd

from src.contracts import BacktestResult


class RegimeAnalyzer:
    """Compare performance across externally supplied regime labels.

    Invariants:
        Regime labels are supplied by the caller; this component never detects,
        defines, or infers regimes autonomously.
        Regime analysis consumes existing ``BacktestResult`` values and does
        not rerun engine, validation, reporting, or UI logic.
    """

    def compare_by_regime(
        self,
        backtest_result: BacktestResult,
        regime_labels: pd.Series | None,
    ) -> dict[str, float]:
        """Compare net-return performance across external regimes.

        Args:
            backtest_result: Canonical engine output containing ``net_returns``.
            regime_labels: Externally supplied labels aligned with
                ``backtest_result.net_returns``.

        Returns:
            Mapping from regime label to numeric performance summary.

        Raises:
            ValueError: If ``regime_labels`` is ``None``, empty, misaligned, or
                otherwise not supplied externally.

        Invariants:
            Does not define or detect regime labels.
            Does not recalculate strategy signals, costs, validation, or
            reporting output.
        """
        if regime_labels is None or regime_labels.empty:
            raise ValueError("external regime labels are required")

        labels = regime_labels.reindex(backtest_result.net_returns.index)
        if labels.isna().any():
            raise ValueError("regime labels must align with net returns")

        returns = backtest_result.net_returns.astype("float64")
        regime_metrics: dict[str, float] = {}
        for label in labels.unique():
            regime_returns = returns[labels == label]
            regime_metrics[str(label)] = float(regime_returns.mean())
        return regime_metrics
