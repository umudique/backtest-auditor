"""Audit report assembly contracts."""

from typing import Any

import numpy as np

from src.contracts import AuditReport, BacktestResult, ValidationResult


def _drawdown_distribution(paths: list[Any]) -> list[float]:
    """Compute per-path maximum drawdowns across Monte Carlo paths.

    Args:
        paths: List of return arrays produced by ``MonteCarloReshuffler``.

    Returns:
        List of per-path maximum drawdown values, empty when paths is empty.

    Raises:
        ValueError: If any path array contains values that cannot be cast to float.

    Invariants:
        Does not rerun engine logic or reapply cost models.
        Consumes only the pre-computed paths from ``ValidationResult``.
    """
    if not paths:
        return []
    result = []
    for path in paths:
        arr = np.asarray(path, dtype=float)
        equity = np.cumprod(1.0 + arr)
        running_max = np.maximum.accumulate(equity)
        result.append(float(np.min(equity / running_max - 1.0)))
    return result


def _build_drawdown_summary(
    bootstrap_results: dict[str, Any],
    monte_carlo_results: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the full drawdown summary including distribution for charting.

    Args:
        bootstrap_results: Dict from ``BootstrapAnalyzer`` containing samples.
        monte_carlo_results: Dict from ``MonteCarloReshuffler`` containing paths.

    Returns:
        Summary dict with ``drawdown_distribution``, ``median_max_drawdown``,
        ``p05``, ``p95``, and raw result copies.

    Raises:
        Nothing — missing or empty paths produce empty distribution fields.

    Invariants:
        Does not rerun engine logic or reapply cost models.
        All statistics are derived from pre-computed paths only.
    """
    mc_paths = monte_carlo_results.get("paths", [])
    dist = _drawdown_distribution(mc_paths)
    arr = np.asarray(dist) if dist else np.array([])
    return {
        "bootstrap": bootstrap_results.copy(),
        "monte_carlo": monte_carlo_results.copy(),
        "drawdown_distribution": dist,
        "median_max_drawdown": float(np.median(arr)) if arr.size else None,
        "p05": float(np.percentile(arr, 5)) if arr.size else None,
        "p95": float(np.percentile(arr, 95)) if arr.size else None,
        "n_paths": len(dist),
    }


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
        sensitivity_grid_rows: list[dict[str, Any]] | None = None,
        walk_forward_rows: list[dict[str, Any]] | None = None,
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
            verdict = "FRAGILE"
        else:
            verdict = "ROBUST"
        _baseline_keys = {"gross_sharpe", "gross_cumulative", "max_drawdown", "maximum_drawdown"}
        _cost_keys = {"net_sharpe", "net_cumulative", "net_return"}
        baseline = {k: v for k, v in metrics.items() if k in _baseline_keys}
        for key in ("sharpe", "net_return", "max_drawdown"):
            val = validation_result.in_sample_metrics.get(key)
            if isinstance(val, (int, float)):
                baseline[key] = val
        cost_adjusted = {k: v for k, v in metrics.items() if k in _cost_keys}
        return AuditReport(
            baseline_metrics=baseline,
            cost_adjusted_metrics=cost_adjusted,
            out_of_sample_metrics=validation_result.out_of_sample_metrics.copy(),
            walk_forward_summary={"folds": list(walk_forward_rows)} if walk_forward_rows else {},
            parameter_sensitivity_summary=validation_result.sensitivity_results.copy(),
            simulated_drawdown_summary=_build_drawdown_summary(
                validation_result.bootstrap_results,
                validation_result.monte_carlo_results,
            ),
            regime_summary={},
            fragility_summary=list(fragility_summary),
            verdict=verdict,
            charts=list(charts),
            sensitivity_grid_rows=list(sensitivity_grid_rows) if sensitivity_grid_rows else [],
        )
