"""Reporting view-model contracts."""

from typing import Any

from src.contracts import BacktestResult, ValidationResult


class EquityCurveViewModel:
    """Prepare equity-curve data for visualization.

    Invariants:
        View models adapt already-computed analytical outputs for reporting
        only.
        They do not calculate metrics, costs, validation evidence, fragility
        conclusions, or verdicts.
    """

    def build(self, backtest_result: BacktestResult) -> dict[str, Any]:
        """Build equity-curve view data.

        Args:
            backtest_result: Canonical engine output containing equity evidence.

        Returns:
            Chart-ready equity-curve data.

        Raises:
            NotImplementedError: Until Phase 2 supplies view-model assembly.
            ValueError: If required equity evidence is absent.

        Invariants:
            Uses existing ``BacktestResult.equity_curve`` only.
            Does not mutate source analytical outputs.
        """
        return {"equity_curve": backtest_result.equity_curve.to_dict()}


class DrawdownViewModel:
    """Prepare drawdown data for visualization."""

    def build(self, backtest_result: BacktestResult) -> dict[str, Any]:
        """Build drawdown view data.

        Args:
            backtest_result: Canonical engine output containing drawdown
                evidence.

        Returns:
            Chart-ready drawdown data.

        Raises:
            NotImplementedError: Until Phase 2 supplies view-model assembly.
            ValueError: If required drawdown evidence is absent.

        Invariants:
            Uses existing ``BacktestResult.drawdown_series`` only.
            Does not recalculate drawdown values.
        """
        return {"drawdown": backtest_result.drawdown_series.to_dict()}


class SensitivityHeatmapViewModel:
    """Prepare parameter-sensitivity evidence for heatmap visualization."""

    def build(self, validation_result: ValidationResult) -> dict[str, Any]:
        """Build sensitivity heatmap view data.

        Args:
            validation_result: Canonical validation output containing
                diagnostic sensitivity results.

        Returns:
            Heatmap-ready parameter-sensitivity data.

        Raises:
            NotImplementedError: Until Phase 2 supplies view-model assembly.
            ValueError: If sensitivity evidence is absent or implies parameter
                selection, ranking, or recommendation.

        Invariants:
            Sensitivity is diagnostic only and never optimization.
            No best, recommended, or optimal parameter is produced.
        """
        return {"sensitivity": validation_result.sensitivity_results.copy()}


class WalkForwardViewModel:
    """Prepare walk-forward validation evidence for reporting."""

    def build(self, validation_result: ValidationResult) -> dict[str, Any]:
        """Build walk-forward summary view data.

        Args:
            validation_result: Canonical validation output containing
                walk-forward results.

        Returns:
            Reporting summary for walk-forward windows and outcomes.

        Raises:
            NotImplementedError: Until Phase 2 supplies view-model assembly.
            ValueError: If walk-forward evidence is absent or evaluation
                windows cannot be attributed.

        Invariants:
            Walk-forward evaluation windows remain separately attributable.
            IS/OOS evidence is not merged into a single headline metric.
        """
        return {"results": list(validation_result.walk_forward_results)}


class MonteCarloDrawdownViewModel:
    """Prepare simulated drawdown evidence for reporting."""

    def build(self, validation_result: ValidationResult) -> dict[str, Any]:
        """Build Monte Carlo drawdown view data.

        Args:
            validation_result: Canonical validation output containing Monte
                Carlo or bootstrap evidence.

        Returns:
            Presentation-ready simulated drawdown summary data.

        Raises:
            NotImplementedError: Until Phase 2 supplies view-model assembly.
            ValueError: If stochastic drawdown evidence is absent or lacks
                reproducibility metadata.

        Invariants:
            Stochastic evidence remains tied to recorded random-seed metadata.
            Reporting does not rerun simulations or reshuffle returns.
        """
        return {
            "bootstrap": validation_result.bootstrap_results.copy(),
            "monte_carlo": validation_result.monte_carlo_results.copy(),
        }
