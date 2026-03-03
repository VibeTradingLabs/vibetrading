"""Edge case and robustness tests."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from vibetrading._core.backtest import BacktestEngine
from vibetrading._core.static_sandbox import StaticSandbox


def _make_ohlcv(base_price=50000.0, periods=200):
    rng = np.random.default_rng(42)
    idx = pd.date_range("2025-01-01", periods=periods, freq="1h", tz=timezone.utc)
    close = base_price + np.cumsum(rng.normal(0, 100, periods))
    return pd.DataFrame(
        {
            "open": close - rng.uniform(0, 50, periods),
            "high": close + rng.uniform(0, 100, periods),
            "low": close - rng.uniform(0, 100, periods),
            "close": close,
            "volume": rng.uniform(100, 10000, periods),
            "fundingRate": rng.uniform(-0.001, 0.001, periods),
            "openInterest": rng.uniform(1e6, 1e8, periods),
        },
        index=idx,
    )


def _make_sandbox(**kwargs):
    btc_data = _make_ohlcv()
    data = {
        ("BTC/USDT:USDT", "1h"): btc_data,
        ("BTC/USDT", "1h"): btc_data,
    }
    defaults = dict(
        exchange="binance",
        start_date="2025-01-01",
        end_date="2025-01-05",
        initial_balances={"USDC": 10000.0},
        data=data,
    )
    defaults.update(kwargs)
    sb = StaticSandbox(**defaults)
    sb.set_backtest_interval("1h")
    return sb


class TestSandboxEdgeCases:
    def test_buy_with_insufficient_balance(self):
        sb = _make_sandbox(initial_balances={"USDC": 1.0})
        price = sb.get_price("BTC")
        result = sb.buy("BTC", 1.0, price, order_type="market")
        # Should either fail or execute with reduced size
        # The key is it shouldn't crash
        assert isinstance(result, dict)

    def test_sell_more_than_held(self):
        sb = _make_sandbox(initial_balances={"USDC": 10000.0, "BTC": 0.001})
        price = sb.get_price("BTC")
        result = sb.sell("BTC", 100.0, price, order_type="market")
        assert isinstance(result, dict)

    def test_reduce_more_than_position(self):
        sb = _make_sandbox()
        sb.set_leverage("BTC", 3)
        price = sb.get_price("BTC")
        sb.long("BTC", 0.01, price, order_type="market")
        result = sb.reduce_position("BTC", 100.0)  # Way more than position
        assert isinstance(result, dict)

    def test_set_leverage_then_change(self):
        sb = _make_sandbox()
        sb.set_leverage("BTC", 3)
        sb.set_leverage("BTC", 5)
        assert sb.futures_position_details["BTC"]["leverage"] == 5

    def test_multiple_positions_different_assets(self):
        eth_data = _make_ohlcv(base_price=3000)
        data = {
            ("BTC/USDT:USDT", "1h"): _make_ohlcv(),
            ("BTC/USDT", "1h"): _make_ohlcv(),
            ("ETH/USDT:USDT", "1h"): eth_data,
            ("ETH/USDT", "1h"): eth_data,
        }
        sb = StaticSandbox(
            exchange="binance",
            start_date="2025-01-01",
            end_date="2025-01-05",
            initial_balances={"USDC": 100000.0},
            data=data,
        )
        sb.set_backtest_interval("1h")

        sb.set_leverage("BTC", 3)
        sb.set_leverage("ETH", 3)

        btc_price = sb.get_price("BTC")
        eth_price = sb.get_price("ETH")

        sb.long("BTC", 0.01, btc_price, order_type="market")
        sb.short("ETH", 0.1, eth_price, order_type="market")

        btc_pos = sb.get_perp_position("BTC")
        eth_pos = sb.get_perp_position("ETH")

        assert btc_pos is not None
        assert eth_pos is not None
        assert btc_pos["size"] > 0  # long
        assert eth_pos["size"] < 0  # short

    def test_rapid_open_close_cycles(self):
        sb = _make_sandbox()
        sb.set_leverage("BTC", 3)
        price = sb.get_price("BTC")

        for _ in range(10):
            sb.long("BTC", 0.01, price, order_type="market")
            sb.reduce_position("BTC", 0.01)

        pos = sb.get_perp_position("BTC")
        assert pos is None or abs(pos.get("size", 0)) < 0.001


class TestBacktestEdgeCases:
    def test_strategy_with_no_trades(self):
        """Strategy that never enters a position."""
        code = """
import math
from vibetrading import vibe, get_perp_price

@vibe
def strategy():
    price = get_perp_price("BTC")
    # Never trade
    pass
"""
        data = {"BTC/1h": _make_ohlcv()}
        engine = BacktestEngine(
            interval="1h",
            data=data,
            start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 3, tzinfo=timezone.utc),
        )
        result = engine.run(code)
        assert result is not None
        assert result["total_trades"] == 0
        assert result["metrics"]["total_return"] == 0.0

    def test_strategy_with_syntax_error_raises(self):
        code = """
from vibetrading import vibe

@vibe
def strategy(
    # Missing closing paren
"""
        data = {"BTC/1h": _make_ohlcv()}
        engine = BacktestEngine(interval="1h", data=data)
        with pytest.raises(SyntaxError):
            engine.run(code)

    def test_strategy_with_runtime_error_raises(self):
        code = """
from vibetrading import vibe

@vibe
def strategy():
    x = 1 / 0
"""
        data = {"BTC/1h": _make_ohlcv()}
        engine = BacktestEngine(
            interval="1h",
            data=data,
            start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 2, tzinfo=timezone.utc),
        )
        with pytest.raises(ZeroDivisionError):
            engine.run(code)

    def test_very_short_backtest(self):
        """Backtest with just a few steps."""
        code = """
from vibetrading import vibe, get_perp_price

@vibe
def strategy():
    pass
"""
        data = {"BTC/1h": _make_ohlcv()}
        engine = BacktestEngine(
            interval="1h",
            data=data,
            initial_balances={"USDC": 10000},
            start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 3, 0, tzinfo=timezone.utc),
        )
        result = engine.run(code)
        assert result is not None
        assert result["simulation_info"]["steps"] <= 4

    def test_small_balance_backtest(self):
        """Backtest with very small balance."""
        code = """
import math
from vibetrading import vibe, get_perp_price, set_leverage, long

@vibe
def strategy():
    price = get_perp_price("BTC")
    if math.isnan(price):
        return
    set_leverage("BTC", 3)
    long("BTC", 0.0001, price, order_type="market")
"""
        data = {"BTC/1h": _make_ohlcv()}
        engine = BacktestEngine(
            interval="1h",
            data=data,
            initial_balances={"USDC": 10},
            start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 2, tzinfo=timezone.utc),
        )
        result = engine.run(code)
        assert result is not None


class TestEquityCurveEdgeCases:
    def test_empty_engine_equity_curve(self):
        data = {"BTC/1h": _make_ohlcv()}
        engine = BacktestEngine(interval="1h", data=data)
        ec = engine.get_equity_curve()
        assert len(ec) == 0

    def test_equity_curve_cumulative_starts_at_zero(self):
        code = """
from vibetrading import vibe, get_perp_price

@vibe
def strategy():
    pass
"""
        data = {"BTC/1h": _make_ohlcv()}
        engine = BacktestEngine(
            interval="1h",
            data=data,
            start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 3, tzinfo=timezone.utc),
        )
        engine.run(code)
        ec = engine.get_equity_curve()
        # First cumulative return should be 0 (no change from initial)
        assert abs(ec["cumulative_returns"].iloc[0]) < 1e-10
