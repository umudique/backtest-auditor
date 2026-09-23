"""Validation result aggregation contracts."""

from typing import Any

from src.contracts import ValidationResult


class ValidationResultAggregator:
    """Normalize validation outputs into canonical ``ValidationResult``.

    Invariants:
        In-sample and out-of-sample metrics are always separate fields.
        IS/OOS metrics are never averaged, merged, or collapsed into a single
        headline metric.
        Aggregation does not calculate strategy logic, costs, metrics,
        reporting, or UI output.
    """

    def aggregate(
        self,
        in_sample_metrics: dict[str, Any],
        out_of_sample_metrics: dict[str, Any],
        walk_forward_results: list[dict[str, Any]],
        sensitivity_results: dict[str, Any],
        bootstrap_results: dict[str, Any],
        monte_carlo_results: dict[str, Any],
    ) -> ValidationResult:
        """Build a canonical validation result from separate evidence.

        Args:
            in_sample_metrics: Metrics or diagnostics for the in-sample segment.
            out_of_sample_metrics: Metrics or diagnostics for the OOS segment.
            walk_forward_results: Walk-forward validation windows and outcomes.
            sensitivity_results: Diagnostic sensitivity grid results.
            bootstrap_results: Seeded bootstrap outputs.
            monte_carlo_results: Seeded Monte Carlo outputs.

        Returns:
            Canonical ``ValidationResult`` with separate
            ``in_sample_metrics`` and ``out_of_sample_metrics`` fields.

        Raises:
            ValueError: If IS/OOS inputs are missing, merged, or aliased to the
                same object, or if sensitivity output implies a recommended
                parameter set.

        Invariants:
            The returned object never contains a merged IS/OOS metric field.
            Validation evidence remains attributable to the procedure that
            produced it.
            This method does not recalculate engine results or metrics.
        """
        if in_sample_metrics is out_of_sample_metrics:
            raise ValueError("in_sample_metrics and out_of_sample_metrics must be separate")

        forbidden_keys = {"best", "recommended", "optimal"}
        if forbidden_keys.intersection(sensitivity_results):
            raise ValueError("sensitivity_results must not recommend parameters")

        return ValidationResult(
            in_sample_metrics=in_sample_metrics.copy(),
            out_of_sample_metrics=out_of_sample_metrics.copy(),
            walk_forward_results=list(walk_forward_results),
            sensitivity_results=sensitivity_results.copy(),
            bootstrap_results=bootstrap_results.copy(),
            monte_carlo_results=monte_carlo_results.copy(),
        )
