"""Chart rendering contracts for reporting."""

from typing import Any

from src.contracts import BacktestResult


class PlotlyRenderer:
    """Render report charts from canonical analytical outputs.

    Invariants:
        Rendering prepares chart objects only.
        Rendering does not calculate returns, drawdowns, metrics, validation
        evidence, fragility conclusions, or verdicts.
        Rendering must not mutate the supplied ``BacktestResult``.
    """

    def render_equity_curve(self, backtest_result: BacktestResult) -> Any:
        """Render an equity-curve chart object.

        Args:
            backtest_result: Canonical engine output containing an
                ``equity_curve`` series that has already been calculated by the
                engine.

        Returns:
            A non-``None`` chart object suitable for downstream presentation.

        Raises:
            NotImplementedError: Until Phase 2 supplies chart construction.
            ValueError: If required equity-curve evidence is missing or if
                rendering would need to calculate analytical values.

        Invariants:
            Uses existing equity-curve data only.
            Does not mutate ``backtest_result``.
            Does not perform reporting verdict or fragility evaluation.
        """
        return {"kind": "equity_curve", "data": backtest_result.equity_curve.to_dict()}

    def render_drawdown(self, backtest_result: BacktestResult) -> Any:
        """Render a drawdown chart object.

        Args:
            backtest_result: Canonical engine output containing a
                ``drawdown_series`` that has already been calculated upstream.

        Returns:
            A non-``None`` chart object suitable for downstream presentation.

        Raises:
            NotImplementedError: Until Phase 2 supplies chart construction.
            ValueError: If required drawdown evidence is missing or if
                rendering would need to calculate analytical values.

        Invariants:
            Uses existing drawdown-series data only.
            Does not mutate ``backtest_result``.
            Does not recalculate drawdowns from returns.
        """
        return {"kind": "drawdown", "data": backtest_result.drawdown_series.to_dict()}
