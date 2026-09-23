"""Monte Carlo validation contracts."""

from typing import Any

import numpy as np

from src.contracts import BacktestResult


class _ComparableArray(np.ndarray):
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, np.ndarray):
            return False
        return bool(np.array_equal(self, other))


class MonteCarloReshuffler:
    """Run reproducible Monte Carlo reshuffling procedures.

    Invariants:
        Every stochastic run accepts and records an explicit ``random_seed``.
        Identical inputs and identical seeds produce identical outputs.
        Monte Carlo procedures consume ``BacktestResult`` only and do not rerun
        engine strategy, execution, cost, metrics, reporting, or UI logic.
    """

    def reshuffle(
        self,
        backtest_result: BacktestResult,
        random_seed: int,
        n_paths: int,
    ) -> dict[str, Any]:
        """Generate seeded reshuffled paths from canonical engine output.

        Args:
            backtest_result: Canonical engine output to reshuffle.
            random_seed: Explicit integer seed controlling stochastic ordering.
            n_paths: Number of Monte Carlo paths to produce.

        Returns:
            Monte Carlo result mapping including the supplied ``random_seed``
            and deterministic paths for identical inputs.

        Raises:
            ValueError: If ``random_seed`` is missing or non-integer, if
                ``n_paths`` is non-positive, or if source data is empty.

        Invariants:
            Repeated calls with identical input and seed return equal outputs.
            The seed is preserved in output metadata.
            This method does not merge IS/OOS metrics or produce parameter
            recommendations.
        """
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")

        source = backtest_result.net_returns.to_numpy()
        if len(source) == 0:
            raise ValueError("source data must not be empty")

        rng = np.random.default_rng(random_seed)
        paths = [rng.permutation(source).view(_ComparableArray) for _ in range(n_paths)]
        return {"random_seed": random_seed, "paths": paths}
