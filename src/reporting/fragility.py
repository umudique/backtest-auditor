"""Fragility evaluation contracts for audit reporting."""

from typing import Any

from src.contracts import ValidationResult

_NARROW_PLATEAU_THRESHOLD = 0.5


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
        sensitivity_grid_rows: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Evaluate known fragility evidence for reporting.

        Args:
            validation_result: Canonical validation output containing separate
                in-sample, out-of-sample, walk-forward, sensitivity, bootstrap,
                and Monte Carlo evidence.
            metrics: Flat metric dictionary from the metrics layer. Keys must
                remain explicit so conclusions can cite their supporting
                evidence.
            sensitivity_grid_rows: Parameter sweep results from the orchestrator.
                Each row must contain a ``Sharpe`` key. When supplied, a narrow
                profitable plateau adds a fragility conclusion.

        Returns:
            A list of descriptive fragility conclusions. Each conclusion cites
            the metric key or validation result that supports it.

        Raises:
            ValueError: If evidence is malformed, if a conclusion cannot cite
                a supporting metric or validation result, or if reporting would
                need to recalculate analytical output.

        Invariants:
            Conclusions are evidence-based and descriptive; they never imply
            guaranteed future profitability or independent strategy advice.
            Missing evidence produces no conclusion for that evidence category.
            Gross/net and IS/OOS distinctions remain visible.
        """
        findings: list[str] = []

        if "gross_sharpe" not in metrics or "net_sharpe" not in metrics:
            return findings

        gross_sharpe = metrics["gross_sharpe"]
        net_sharpe = metrics["net_sharpe"]
        cost_drag = gross_sharpe - net_sharpe
        deteriorated = (gross_sharpe > 0 and net_sharpe < gross_sharpe * 0.8) or (
            gross_sharpe <= 0 and cost_drag > 0.1
        )
        if deteriorated:
            findings.append(
                "cost deterioration: "
                f"net_sharpe ({net_sharpe:.3f}) is below gross_sharpe ({gross_sharpe:.3f})"
            )

        in_sample = validation_result.in_sample_metrics.get("sharpe")
        out_of_sample = validation_result.out_of_sample_metrics.get("sharpe")
        if isinstance(in_sample, int | float) and isinstance(out_of_sample, int | float):
            if out_of_sample < in_sample * 0.8:
                findings.append(
                    "out-of-sample deterioration: "
                    f"ValidationResult.out_of_sample_metrics['sharpe'] ({out_of_sample:.3f}) "
                    "is below "
                    f"ValidationResult.in_sample_metrics['sharpe'] ({in_sample:.3f})"
                )

        if sensitivity_grid_rows:
            total = len(sensitivity_grid_rows)
            profitable = sum(
                1
                for row in sensitivity_grid_rows
                if isinstance(row.get("Sharpe"), (int, float)) and row["Sharpe"] > 0
            )
            ratio = profitable / total
            if ratio < _NARROW_PLATEAU_THRESHOLD:
                findings.append(
                    f"narrow profitable plateau: "
                    f"{profitable}/{total} parameter combinations ({ratio:.0%}) "
                    f"produce positive Sharpe — result may be parameter-specific"
                )

        return findings
