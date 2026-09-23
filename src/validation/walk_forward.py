"""Walk-forward validation contracts."""

from src.contracts import BacktestResult


class WalkForwardValidator:
    """Construct chronological train/evaluate validation windows.

    Invariants:
        Evaluation windows are non-overlapping.
        Training windows may overlap when configuration requests it.
        OOS/evaluation windows are not used for parameter fitting.
        The validator consumes ``BacktestResult`` only and does not rerun or
        reimplement engine strategy, cost, or execution logic.
    """

    def construct_windows(
        self,
        backtest_result: BacktestResult,
        train_size: int,
        evaluation_size: int,
        step_size: int,
    ) -> list[dict[str, int]]:
        """Construct walk-forward training and evaluation windows.

        Args:
            backtest_result: Canonical engine output whose chronological index
                defines the available observations.
            train_size: Number of observations in each training window.
            evaluation_size: Number of observations in each evaluation window.
            step_size: Number of observations between consecutive windows.

        Returns:
            List of window descriptors. Each descriptor records training and
            evaluation start/end integer positions.

        Raises:
            ValueError: If sizes are non-positive, if a window would exceed the
                available observations, or if evaluation windows overlap.

        Invariants:
            Evaluation windows are chronological and non-overlapping.
            Training windows never consume their paired evaluation observations.
            Window construction does not calculate metrics, costs, strategy
            signals, reports, or UI output.
        """
        if train_size <= 0 or evaluation_size <= 0 or step_size <= 0:
            raise ValueError("train_size, evaluation_size, and step_size must be positive")

        total_observations = len(backtest_result.net_returns)
        if train_size + evaluation_size > total_observations:
            raise ValueError("train_size plus evaluation_size exceeds available observations")

        windows: list[dict[str, int]] = []
        evaluation_start = train_size
        while evaluation_start + evaluation_size <= total_observations:
            train_end = evaluation_start
            train_start = train_end - train_size
            evaluation_end = evaluation_start + evaluation_size
            windows.append(
                {
                    "train_start": train_start,
                    "train_end": train_end,
                    "evaluation_start": evaluation_start,
                    "evaluation_end": evaluation_end,
                }
            )
            evaluation_start += max(step_size, evaluation_size)

        return windows
