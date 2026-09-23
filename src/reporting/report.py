"""Audit report assembly contracts."""

from typing import Any

from src.contracts import AuditReport, BacktestResult, ValidationResult


class AuditReportBuilder:
    """Assemble canonical audit reports from downstream analytical evidence.

    Invariants:
        This builder consumes existing engine, validation, metrics, fragility,
        and chart outputs only.
        It never recalculates returns, reapplies costs, reruns validation, or
        renders UI.
        The final verdict is derived from ``fragility_summary`` and is never
        hardcoded or supplied by default.
    """

    def build(
        self,
        backtest_result: BacktestResult,
        validation_result: ValidationResult,
        metrics: dict[str, float],
        fragility_summary: list[str],
        charts: list[Any],
    ) -> AuditReport:
        """Build a canonical ``AuditReport``.

        Args:
            backtest_result: Canonical engine output whose execution metadata
                and gross/net distinction remain reportable evidence.
            validation_result: Canonical validation output with separate IS/OOS
                metrics and robustness evidence.
            metrics: Flat metrics dictionary produced upstream by the metrics
                layer.
            fragility_summary: Evidence-based reporting conclusions produced by
                ``FragilityEvaluator``.
            charts: Presentation-ready chart objects or chart view models.

        Returns:
            A canonical ``AuditReport`` from ``src.contracts`` with every
            required report section populated and a verdict derived from
            ``fragility_summary``.

        Raises:
            NotImplementedError: Until Phase 2 supplies report assembly rules.
            ValueError: If ``fragility_summary`` is empty, if a verdict cannot
                be derived from explicit evidence, or if report assembly would
                collapse gross/net or IS/OOS evidence.

        Invariants:
            ``AuditReport.verdict`` has no default and is derived from explicit
            report evidence.
            Report sections preserve baseline, cost-adjusted, OOS,
            walk-forward, sensitivity, simulated drawdown, regime, fragility,
            verdict, and chart fields separately.
            No financial calculations are performed in reporting.
        """
        del backtest_result

        if fragility_summary:
            verdict = "FRAGILE: " + "; ".join(fragility_summary)
        else:
            verdict = "PASS: no significant fragility detected"
        return AuditReport(
            baseline_metrics={key: value for key, value in metrics.items() if "gross" in key},
            cost_adjusted_metrics={key: value for key, value in metrics.items() if "net" in key},
            out_of_sample_metrics=validation_result.out_of_sample_metrics.copy(),
            walk_forward_summary={"results": list(validation_result.walk_forward_results)},
            parameter_sensitivity_summary=validation_result.sensitivity_results.copy(),
            simulated_drawdown_summary={
                "bootstrap": validation_result.bootstrap_results.copy(),
                "monte_carlo": validation_result.monte_carlo_results.copy(),
            },
            regime_summary={},
            fragility_summary=list(fragility_summary),
            verdict=verdict,
            charts=list(charts),
        )
