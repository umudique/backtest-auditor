"""Fragility evaluation contracts for audit reporting."""

from src.contracts import ValidationResult


class FragilityEvaluator:
    """Convert analytical evidence into traceable fragility conclusions.

    Invariants:
        Reporting consumes analytical outputs only and never reruns the engine,
        recalculates returns, reapplies fees, or calculates new performance
        metrics.
        Every emitted conclusion cites the metric key or validation result that
        supports it.
        No conclusion is emitted when the supporting evidence is absent.
    """

    def evaluate(
        self,
        validation_result: ValidationResult,
        metrics: dict[str, float],
    ) -> list[str]:
        """Evaluate known fragility evidence for reporting.

        Args:
            validation_result: Canonical validation output containing separate
                in-sample, out-of-sample, walk-forward, sensitivity, bootstrap,
                and Monte Carlo evidence.
            metrics: Flat metric dictionary from the metrics layer. Keys must
                remain explicit so conclusions can cite their supporting
                evidence.

        Returns:
            A list of descriptive fragility conclusions. Each conclusion cites
            the metric key or validation result that supports it.

        Raises:
            NotImplementedError: Until Phase 2 supplies evidence-to-conclusion
                rules.
            ValueError: If evidence is malformed, if a conclusion cannot cite
                a supporting metric or validation result, or if reporting would
                need to recalculate analytical output.

        Invariants:
            Conclusions are evidence-based and descriptive; they never imply
            guaranteed future profitability or independent strategy advice.
            Missing evidence produces no conclusion for that evidence category.
            Gross/net and IS/OOS distinctions remain visible.
        """
        if "gross_sharpe" not in metrics or "net_sharpe" not in metrics:
            return []

        gross_sharpe = metrics["gross_sharpe"]
        net_sharpe = metrics["net_sharpe"]
        if net_sharpe < gross_sharpe * 0.8:
            return [
                "cost deterioration: "
                f"net_sharpe ({net_sharpe:.3f}) is below gross_sharpe ({gross_sharpe:.3f})"
            ]

        in_sample = validation_result.in_sample_metrics.get("sharpe")
        out_of_sample = validation_result.out_of_sample_metrics.get("sharpe")
        if isinstance(in_sample, int | float) and isinstance(out_of_sample, int | float):
            if out_of_sample < in_sample * 0.8:
                return [
                    "out-of-sample deterioration: "
                    f"ValidationResult.out_of_sample_metrics['sharpe'] ({out_of_sample:.3f}) "
                    "is below "
                    f"ValidationResult.in_sample_metrics['sharpe'] ({in_sample:.3f})"
                ]

        return []
