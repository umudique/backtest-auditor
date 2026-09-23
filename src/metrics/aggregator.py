"""Metrics aggregation contracts."""


class MetricsAggregator:
    """Aggregate metric component outputs into a flat numeric dictionary.

    Invariants:
        Output is a flat ``dict[str, float]`` with no nested structures.
        Aggregation does not recalculate metrics, rerun the engine, run
        validation, produce reports, or invoke UI logic.
    """

    def aggregate(
        self,
        return_metrics: dict[str, float],
        drawdown_metrics: dict[str, float],
        regime_metrics: dict[str, float],
    ) -> dict[str, float]:
        """Combine metric outputs into a canonical flat dictionary.

        Args:
            return_metrics: Flat numeric metrics from return calculations.
            drawdown_metrics: Flat numeric metrics from drawdown calculations.
            regime_metrics: Flat numeric metrics from externally labeled regime
                comparisons.

        Returns:
            Flat ``dict[str, float]`` suitable for downstream reporting.

        Raises:
            ValueError: If any value is non-numeric, any value is nested, or any
                metric key is duplicated ambiguously.

        Invariants:
            No nested dictionaries, lists, tuples, or opaque metric groups are
            returned.
            Aggregation preserves explicit metric keys.
        """
        aggregated: dict[str, float] = {}
        for metric_group in (return_metrics, drawdown_metrics, regime_metrics):
            for key, value in metric_group.items():
                if key in aggregated:
                    raise ValueError(f"duplicate metric key: {key}")
                if not isinstance(value, int | float):
                    raise ValueError(f"metric value for {key} must be numeric")
                aggregated[key] = float(value)
        return aggregated
