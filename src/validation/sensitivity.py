"""Parameter sensitivity validation contracts."""

from itertools import product
from typing import Any

from src.contracts import BacktestResult


class ParameterSensitivityAnalyzer:
    """Measure robustness across a diagnostic parameter grid.

    Invariants:
        Sensitivity is diagnostic only; it never selects, ranks, recommends, or
        optimizes parameter sets.
        Results are represented as a grid of observed outcomes around a fixed
        parameter point.
        The analyzer consumes ``BacktestResult`` only and does not rerun or
        reimplement engine strategy, execution, cost, reporting, or UI logic.
    """

    def analyze(
        self,
        backtest_result: BacktestResult,
        parameter_grid: dict[str, list[Any]],
    ) -> dict[str, Any]:
        """Analyze parameter sensitivity over a diagnostic grid.

        Args:
            backtest_result: Canonical engine output used as the validation
                reference.
            parameter_grid: Diagnostic parameter values to inspect around a
                fixed point. The grid is not a search space.

        Returns:
            Mapping containing a ``grid`` result and supporting diagnostics.
            The output must not contain ``best``, ``recommended``, or
            ``optimal`` keys.

        Raises:
            ValueError: If the grid is empty, malformed, or implies selecting a
                recommended parameter set.

        Invariants:
            No returned field may imply best/recommended/optimal parameters.
            Sensitivity output supports fragility analysis only.
            IS/OOS metrics are not merged by this method.
        """
        if not parameter_grid:
            raise ValueError("parameter_grid must not be empty")

        parameter_names = list(parameter_grid)
        value_lists = [parameter_grid[name] for name in parameter_names]
        if any(not values for values in value_lists):
            raise ValueError("parameter_grid values must not be empty")

        grid = [
            {"params": dict(zip(parameter_names, combination, strict=True)), "result": None}
            for combination in product(*value_lists)
        ]
        return {"grid": grid}
