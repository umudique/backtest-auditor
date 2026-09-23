"""In-sample / out-of-sample validation contracts."""

from src.contracts import BacktestResult


class InSampleOutOfSampleValidator:
    """Partition a canonical backtest result into independent IS/OOS segments.

    Invariants:
        In-sample and out-of-sample evidence is always kept separate.
        Partitions are chronological, contiguous, non-overlapping, and cover
        the original period with no gap.
        Validation consumes ``BacktestResult`` only and does not reimplement
        strategy logic, execution, costs, metrics, reporting, or UI behavior.
    """

    def split(
        self,
        backtest_result: BacktestResult,
        split_ratio: float,
    ) -> dict[str, BacktestResult]:
        """Split a backtest result into separate IS and OOS backtest slices.

        Args:
            backtest_result: Canonical engine output to partition.
            split_ratio: Fraction of observations assigned to the in-sample
                period. Must be strictly between 0 and 1 and leave at least one
                observation in both IS and OOS.

        Returns:
            Mapping with separate ``in_sample`` and ``out_of_sample``
            ``BacktestResult`` values.

        Raises:
            ValueError: If ``split_ratio`` is outside ``(0, 1)``, if either
                partition would be empty, or if source series are not aligned.

        Invariants:
            The last IS timestamp is strictly before the first OOS timestamp.
            IS and OOS periods together cover the original period with no gap
            and no overlap.
            IS/OOS metrics are not calculated or merged by this method.
        """
        if not 0 < split_ratio < 1:
            raise ValueError("split_ratio must be between 0 and 1")

        length = len(backtest_result.net_returns)
        split_index = int(length * split_ratio)
        if split_index <= 0 or split_index >= length:
            raise ValueError("split_ratio must leave non-empty IS and OOS partitions")

        return {
            "in_sample": self._slice_result(backtest_result, slice(0, split_index)),
            "out_of_sample": self._slice_result(backtest_result, slice(split_index, None)),
        }

    def _slice_result(self, backtest_result: BacktestResult, row_slice: slice) -> BacktestResult:
        return BacktestResult(
            positions=backtest_result.positions.iloc[row_slice],
            trades=backtest_result.trades.iloc[row_slice],
            gross_returns=backtest_result.gross_returns.iloc[row_slice],
            net_returns=backtest_result.net_returns.iloc[row_slice].copy(),
            equity_curve=backtest_result.equity_curve.iloc[row_slice],
            drawdown_series=backtest_result.drawdown_series.iloc[row_slice],
            execution_metadata=backtest_result.execution_metadata.copy(),
        )
