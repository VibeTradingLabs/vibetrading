"""Tests for slippage modeling."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from vibetrading._core.static_sandbox import StaticSandbox


def _make_ohlcv(base_price=50000.0):
    rng = np.random.default_rng(42)
    idx = pd.date_range("2025-01-01", periods=100, freq="1h", tz=timezone.utc)
    close = np.full(100, base_price)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 10,
            "low": close - 10,
            "close": close,
            "volume": rng.uniform(100, 10000, 100),
            "fundingRate": np.zeros(100),
            "openInterest": np.ones(100) * 1e7,
        },
        index=idx,
    )


def _make_sandbox(slippage_bps=0.0):
    btc_data = _make_ohlcv()
    data = {
        ("BTC/USDT:USDT", "1h"): btc_data,
        ("BTC/USDT", "1h"): btc_data,
    }
    sb = StaticSandbox(
        exchange="binance",
        start_date="2025-01-01",
        end_date="2025-01-05",
        initial_balances={"USDC": 100000.0},
        data=data,
        slippage_bps=slippage_bps,
    )
    sb.set_backtest_interval("1h")
    return sb


class TestSlippage:
    def test_no_slippage_by_default(self):
        sb = _make_sandbox(slippage_bps=0)
        price_before = sb.get_price("BTC")
        sb.buy("BTC", 0.1, price_before, order_type="market")
        # Execution price should match market price
        assert sb.trades[0]["price"] == price_before

    def test_buy_slippage_increases_price(self):
        sb = _make_sandbox(slippage_bps=10)  # 0.1% slippage
        base_price = sb.get_price("BTC")
        sb.buy("BTC", 0.1, base_price, order_type="market")
        exec_price = sb.trades[0]["price"]
        assert exec_price > base_price
        expected = base_price * (1 + 10 / 10_000)
        assert abs(exec_price - expected) < 0.01

    def test_sell_slippage_decreases_price(self):
        sb = _make_sandbox(slippage_bps=10)
        sb.balances["BTC"] = 1.0
        base_price = sb.get_price("BTC")
        sb.sell("BTC", 0.1, base_price, order_type="market")
        exec_price = sb.trades[0]["price"]
        assert exec_price < base_price

    def test_long_slippage_increases_price(self):
        sb = _make_sandbox(slippage_bps=10)
        sb.set_leverage("BTC", 3)
        base_price = sb.get_price("BTC")
        sb.long("BTC", 0.01, base_price, order_type="market")
        exec_price = sb.trades[0]["price"]
        assert exec_price > base_price

    def test_short_slippage_decreases_price(self):
        sb = _make_sandbox(slippage_bps=10)
        sb.set_leverage("BTC", 3)
        base_price = sb.get_price("BTC")
        sb.short("BTC", 0.01, base_price, order_type="market")
        exec_price = sb.trades[0]["price"]
        assert exec_price < base_price

    def test_limit_orders_not_affected_by_slippage(self):
        sb = _make_sandbox(slippage_bps=50)
        sb.set_leverage("BTC", 3)
        # Limit order at exact price — slippage should not apply
        sb.long("BTC", 0.01, 49000.0, order_type="limit")
        orders = sb.get_perp_open_orders("BTC")
        assert len(orders) == 1
        assert orders[0]["price"] == 49000.0

    def test_slippage_makes_backtest_more_conservative(self):
        """A backtest with slippage should have lower returns than without."""
        from vibetrading._core.backtest import BacktestEngine

        ohlcv = _make_ohlcv(base_price=50000.0)
        data = {"BTC/1h": ohlcv}

        strategy = """
import math
from vibetrading import vibe, get_perp_price, set_leverage, long, reduce_position, get_perp_position

@vibe
def strategy():
    price = get_perp_price("BTC")
    if math.isnan(price):
        return
    pos = get_perp_position("BTC")
    if not pos:
        set_leverage("BTC", 3)
        long("BTC", 0.01, price, order_type="market")
"""
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 3, tzinfo=timezone.utc)

        r_no_slip = BacktestEngine(
            interval="1h",
            data=data,
            slippage_bps=0,
            initial_balances={"USDC": 10000},
            start_time=start,
            end_time=end,
        ).run(strategy)

        r_with_slip = BacktestEngine(
            interval="1h",
            data=data,
            slippage_bps=50,
            initial_balances={"USDC": 10000},
            start_time=start,
            end_time=end,
        ).run(strategy)

        # With slippage, total value should be lower (paid more for entry)
        assert r_with_slip["metrics"]["total_value"] <= r_no_slip["metrics"]["total_value"]
