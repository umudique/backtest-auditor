"""Phase 1 tests for backtest engine contracts."""

from __future__ import annotations

import pandas as pd
import pytest

from src.contracts import AuditConfig, BacktestResult, MarketData
from src.engine.costs import CostModel
from src.engine.execution import ExecutionModel
from src.engine.position_sizing import PositionSizer
from src.engine.result import BacktestResultBuilder
from src.engine.simulator import PortfolioSimulator
from src.engine.strategy import MovingAverageCrossoverStrategy, StrategyInterface


def market_data_fixture() -> MarketData:
    index = pd.date_range("2024-01-02 09:30", periods=6, freq="min")
    return MarketData(
        timestamp=pd.Series(index, index=index, name="timestamp"),
        open=pd.Series([100.0, 101.0, 102.0, 101.0, 103.0, 104.0], index=index),
        high=pd.Series([101.0, 102.0, 103.0, 102.0, 104.0, 105.0], index=index),
        low=pd.Series([99.0, 100.0, 101.0, 100.0, 102.0, 103.0], index=index),
        close=pd.Series([100.0, 101.0, 102.0, 101.0, 103.0, 104.0], index=index),
        volume=pd.Series([1000, 1100, 1200, 1300, 1400, 1500], index=index),
    )


def audit_config_fixture() -> AuditConfig:
    return AuditConfig(
        strategy_name="moving_average_crossover",
        strategy_parameters={"short_window": 2, "long_window": 3},
        fee_assumption={"fee_rate": 0.001},
        slippage_assumption={"slippage_rate": 0.002},
        position_sizing={"min_position": -1.0, "max_position": 1.0},
        execution_assumptions={"timing": "signal_close_execute_next_open"},
        validation_configuration={},
        random_seed=42,
    )


def test_strategy_interface_bare_generate_signals_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        StrategyInterface().generate_signals(market_data_fixture(), {"short_window": 2})


def test_moving_average_crossover_signals_have_expected_length_and_domain() -> None:
    market_data = market_data_fixture()

    signals = MovingAverageCrossoverStrategy().generate_signals(
        market_data,
        {"short_window": 2, "long_window": 3},
    )

    assert len(signals) == len(market_data.close)
    assert set(signals.dropna().unique()).issubset({-1, 0, 1})


def test_moving_average_crossover_signal_at_t_uses_only_prices_up_to_t() -> None:
    market_data = market_data_fixture()
    strategy = MovingAverageCrossoverStrategy()
    parameters = {"short_window": 2, "long_window": 3}

    full_signals = strategy.generate_signals(market_data, parameters)

    for position in range(len(market_data.close)):
        prefix_index = market_data.close.index[: position + 1]
        prefix_data = MarketData(
            timestamp=market_data.timestamp.loc[prefix_index],
            open=market_data.open.loc[prefix_index],
            high=market_data.high.loc[prefix_index],
            low=market_data.low.loc[prefix_index],
            close=market_data.close.loc[prefix_index],
            volume=market_data.volume.loc[prefix_index],
        )
        prefix_signals = strategy.generate_signals(prefix_data, parameters)

        full_val = full_signals.iloc[position]
        prefix_val = prefix_signals.iloc[-1]
        if pd.isna(full_val):
            assert pd.isna(prefix_val)
        else:
            assert full_val == prefix_val


def test_position_sizer_outputs_bounded_positions_with_signal_length() -> None:
    signals = pd.Series([-1, 0, 1, 1, -1], dtype=float)

    positions = PositionSizer().size_positions(
        signals,
        {"min_position": -0.5, "max_position": 0.75},
    )

    assert len(positions) == len(signals)
    assert positions.between(-0.5, 0.75).all()


def test_execution_model_fills_signal_at_next_bar_open_and_records_metadata() -> None:
    signals = pd.Series([0, 1, -1], index=pd.date_range("2024-01-02", periods=3, freq="D"))
    open_prices = pd.Series([100.0, 101.0, 102.0], index=signals.index)
    config = audit_config_fixture()

    executed_positions, metadata = ExecutionModel().apply_execution_timing(
        signals,
        open_prices,
        config.execution_assumptions,
    )

    assert executed_positions.iloc[1] == signals.iloc[0]
    assert executed_positions.iloc[2] == signals.iloc[1]
    assert metadata["timing"] == "signal_close_execute_next_open"


def test_cost_model_applies_known_fee_and_slippage_to_gross_return() -> None:
    gross_returns = pd.Series([0.05])
    trades = pd.DataFrame({"trade_size": [2.0]})

    net_returns = CostModel().apply_costs(
        gross_returns,
        trades,
        {"fee_rate": 0.001, "slippage_rate": 0.002},
    )

    assert net_returns.iloc[0] == pytest.approx(0.05 - (2.0 * (0.001 + 0.002)))


def test_cost_model_zero_cost_must_be_explicit_and_preserves_gross_value() -> None:
    gross_returns = pd.Series([0.01, -0.02])
    trades = pd.DataFrame({"trade_size": [1.0, 1.0]})

    net_returns = CostModel().apply_costs(
        gross_returns,
        trades,
        {"fee_rate": 0.0, "slippage_rate": 0.0},
    )

    assert net_returns.equals(gross_returns)
    assert net_returns is not gross_returns


def test_cost_model_rejects_missing_cost_configuration() -> None:
    with pytest.raises(ValueError, match="fee|slippage|explicit"):
        CostModel().apply_costs(pd.Series([0.01]), pd.DataFrame({"trade_size": [1.0]}), {})


def test_portfolio_simulator_outputs_normalized_equity_drawdown_and_positions() -> None:
    market_data = market_data_fixture()
    positions = pd.Series([0, 1, 1, 0, -1, 0], index=market_data.close.index)

    simulation = PortfolioSimulator().simulate(
        market_data,
        positions,
        {"initial_equity": 1.0},
    )

    assert simulation["equity_curve"].iloc[0] == pytest.approx(1.0)
    assert (simulation["drawdown_series"] <= 0).all()
    assert len(simulation["positions"]) == len(market_data.close)


def test_backtest_result_builder_emits_canonical_result_with_metadata() -> None:
    index = pd.date_range("2024-01-02", periods=3, freq="D")
    gross_returns = pd.Series([0.0, 0.02, -0.01], index=index)
    net_returns = pd.Series([0.0, 0.018, -0.012], index=index)

    result = BacktestResultBuilder().build(
        positions=pd.Series([0, 1, 1], index=index),
        trades=pd.DataFrame({"trade_size": [0.0, 1.0, 0.0]}, index=index),
        gross_returns=gross_returns,
        net_returns=net_returns,
        equity_curve=pd.Series([1.0, 1.018, 1.005784], index=index),
        drawdown_series=pd.Series([0.0, 0.0, -0.012], index=index),
        execution_metadata={"timing": "signal_close_execute_next_open"},
    )

    assert isinstance(result, BacktestResult)
    assert result.gross_returns is not result.net_returns
    assert result.execution_metadata


def test_gross_vs_net_consistency_for_positive_cost_trades() -> None:
    gross_returns = pd.Series([0.01, 0.02, -0.01])
    trades = pd.DataFrame({"trade_size": [0.0, 1.0, 2.0]})

    net_returns = CostModel().apply_costs(
        gross_returns,
        trades,
        {"fee_rate": 0.001, "slippage_rate": 0.001},
    )

    traded = trades["trade_size"] > 0
    assert (net_returns.loc[traded] <= gross_returns.loc[traded]).all()


def test_engine_components_accept_canonical_contract_types_only() -> None:
    market_data = market_data_fixture()
    config = audit_config_fixture()

    assert isinstance(market_data, MarketData)
    assert isinstance(config, AuditConfig)
