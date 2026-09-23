"""Bootstrap validation contracts."""

from typing import Any

import numpy as np

from src.contracts import BacktestResult


class _ComparableArray(np.ndarray):
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, np.ndarray):
            return False
        return bool(np.array_equal(self, other))


class BootstrapAnalyzer:
    """Run reproducible bootstrap robustness procedures.

    Invariants:
        Every stochastic run accepts and records an explicit ``random_seed``.
        Identical inputs and identical seeds produce identical outputs.
        Bootstrap procedures consume ``BacktestResult`` only and do not rerun
        engine strategy, execution, cost, metrics, reporting, or UI logic.
    """

    def run(
        self,
        backtest_result: BacktestResult,
        random_seed: int,
        n_samples: int,
    ) -> dict[str, Any]:
        """Run a seeded bootstrap analysis over engine output.

        Args:
            backtest_result: Canonical engine output to resample.
            random_seed: Explicit integer seed controlling stochastic sampling.
            n_samples: Number of bootstrap samples to generate.

        Returns:
            Bootstrap result mapping including the supplied ``random_seed`` and
            deterministic sampled outputs for identical inputs.

        Raises:
            ValueError: If ``random_seed`` is missing or non-integer, if
                ``n_samples`` is non-positive, or if source data is empty.

        Invariants:
            Repeated calls with identical input and seed return equal outputs.
            The seed is preserved in output metadata.
            This method does not merge IS/OOS metrics or produce parameter
            recommendations.
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        source = backtest_result.net_returns.to_numpy()
        if len(source) == 0:
            raise ValueError("source data must not be empty")

        rng = np.random.default_rng(random_seed)
        samples = [
            rng.choice(source, size=len(source), replace=True).view(_ComparableArray)
            for _ in range(n_samples)
        ]
        return {"random_seed": random_seed, "samples": samples}
