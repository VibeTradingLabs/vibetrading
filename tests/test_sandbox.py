"""Tests for the StaticSandbox backtesting environment."""

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from vibetrading._core.static_sandbox import StaticSandbox


def _make_ohlcv_data(
    start: str = "2025-01-01",
    periods: int = 100,
    freq: str = "1h",
    base_price: float = 50000.0,
) -> pd.DataFrame:
    """Create synthetic OHLCV data for testing."""
    rng = np.random.default_rng(42)
    idx = pd.date_range(start, periods=periods, freq=freq, tz=timezone.utc)
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


def _make_sandbox(
    initial_balances: dict | None = None,
    data: dict | None = None,
) -> StaticSandbox:
    """Create a sandbox with injected test data."""
    btc_data = _make_ohlcv_data()
    default_data = {
        ("BTC/USDT:USDT", "1h"): btc_data,
        ("BTC/USDT", "1h"): btc_data,
    }
    if data:
        default_data.update(data)

    sb = StaticSandbox(
        exchange="binance",
        start_date="2025-01-01",
        end_date="2025-01-05",
        initial_balances=initial_balances or {"USDC": 10000.0},
        data=default_data,
    )
    sb.set_backtest_interval("1h")
    return sb


class TestSandboxInit:
    def test_default_balances(self):
        sb = _make_sandbox()
        assert sb.balances["USDC"] == 10000.0

    def test_custom_balances(self):
        sb = _make_sandbox(initial_balances={"USDC": 5000.0, "BTC": 0.1})
        assert sb.balances["USDC"] == 5000.0
        assert sb.balances["BTC"] == 0.1

    def test_supported_assets(self):
        sb = _make_sandbox()
        assets = sb.get_supported_assets()
        assert "BTC" in assets
        assert "ETH" in assets

    def test_current_time_set(self):
        sb = _make_sandbox()
        assert sb.current_time == datetime(2025, 1, 1, tzinfo=timezone.utc)


class TestPriceAccess:
    def test_get_price_returns_float(self):
        sb = _make_sandbox()
        price = sb.get_price("BTC")
        assert isinstance(price, float)
        assert price > 0

    def test_get_spot_price(self):
        sb = _make_sandbox()
        price = sb.get_spot_price("BTC")
        assert price > 0

    def test_get_perp_price(self):
        sb = _make_sandbox()
        price = sb.get_perp_price("BTC")
        assert price > 0

    def test_stablecoin_price_is_one(self):
        sb = _make_sandbox()
        assert sb.get_price("USDC") == 1.0
        assert sb.get_price("USDT") == 1.0

    def test_unsupported_asset_returns_nan(self):
        sb = _make_sandbox()
        # An asset with no data loaded
        price = sb.get_price("ZZZZZ")
        assert math.isnan(price)


class TestOHLCV:
    def test_get_futures_ohlcv(self):
        sb = _make_sandbox()
        df = sb.get_futures_ohlcv("BTC", "1h", 10)
        assert len(df) == 1  # Only 1 candle at start time
        assert "close" in df.columns
        assert "fundingRate" in df.columns

    def test_get_futures_ohlcv_columns(self):
        sb = _make_sandbox()
        # Advance time to have more data available
        for _ in range(20):
            sb.advance_time("1h")
        df = sb.get_futures_ohlcv("BTC", "1h", 10)
        expected_cols = ["open", "high", "low", "close", "volume", "fundingRate", "openInterest"]
        for col in expected_cols:
            assert col in df.columns

    def test_get_spot_ohlcv(self):
        sb = _make_sandbox()
        df = sb.get_spot_ohlcv("BTC", "1h", 5)
        assert "close" in df.columns


class TestFundingRate:
    def test_get_funding_rate(self):
        sb = _make_sandbox()
        fr = sb.get_funding_rate("BTC")
        assert isinstance(fr, float)

    def test_get_funding_rate_history(self):
        sb = _make_sandbox()
        for _ in range(10):
            sb.advance_time("1h")
        df = sb.get_funding_rate_history("BTC", 5)
        assert "fundingRate" in df.columns


class TestSpotTrading:
    def test_buy_market_order(self):
        sb = _make_sandbox()
        price = sb.get_price("BTC")
        qty = 0.01
        result = sb.buy("BTC", qty, price, order_type="market")
        assert result["status"] == "success"
        assert sb.balances["BTC"] > 0
        assert sb.balances["USDC"] < 10000.0

    def test_sell_market_order(self):
        sb = _make_sandbox(initial_balances={"USDC": 10000.0, "BTC": 1.0})
        price = sb.get_price("BTC")
        result = sb.sell("BTC", 0.1, price, order_type="market")
        assert result["status"] == "success"
        assert sb.balances["BTC"] < 1.0

    def test_cannot_buy_usdc(self):
        sb = _make_sandbox()
        result = sb.buy("USDC", 100, 1.0)
        assert result["status"] == "error"

    def test_cannot_sell_usdc(self):
        sb = _make_sandbox()
        result = sb.sell("USDC", 100, 1.0)
        assert result["status"] == "error"

    def test_buy_zero_quantity_error(self):
        sb = _make_sandbox()
        result = sb.buy("BTC", 0, 50000)
        assert result["status"] == "error"

    def test_trades_recorded(self):
        sb = _make_sandbox()
        price = sb.get_price("BTC")
        sb.buy("BTC", 0.01, price, order_type="market")
        assert len(sb.trades) == 1
        assert sb.trades[0]["action"] == "buy"
        assert sb.trades[0]["type"] == "spot"


class TestFuturesTrading:
    def test_set_leverage(self):
        sb = _make_sandbox()
        sb.set_leverage("BTC", 5)
        assert sb.futures_position_details["BTC"]["leverage"] == 5

    def test_invalid_leverage_raises(self):
        sb = _make_sandbox()
        with pytest.raises(ValueError):
            sb.set_leverage("BTC", 99)

    def test_long_market_order(self):
        sb = _make_sandbox()
        sb.set_leverage("BTC", 3)
        price = sb.get_price("BTC")
        qty = 0.01
        result = sb.long("BTC", qty, price, order_type="market")
        assert result["status"] == "success"
        pos = sb.get_futures_position("BTC")
        assert pos > 0

    def test_short_market_order(self):
        sb = _make_sandbox()
        sb.set_leverage("BTC", 3)
        price = sb.get_price("BTC")
        result = sb.short("BTC", 0.01, price, order_type="market")
        assert result["status"] == "success"
        pos = sb.get_futures_position("BTC")
        assert pos < 0

    def test_reduce_position(self):
        sb = _make_sandbox()
        sb.set_leverage("BTC", 3)
        price = sb.get_price("BTC")
        sb.long("BTC", 0.1, price, order_type="market")
        result = sb.reduce_position("BTC", 0.05)
        assert result["status"] == "success"
        pos = sb.get_futures_position("BTC")
        assert 0 < pos < 0.1

    def test_reduce_position_full_close(self):
        sb = _make_sandbox()
        sb.set_leverage("BTC", 3)
        price = sb.get_price("BTC")
        sb.long("BTC", 0.1, price, order_type="market")
        sb.reduce_position("BTC", 0.1)
        pos = sb.get_futures_position("BTC")
        assert abs(pos) < 1e-8

    def test_reduce_no_position_error(self):
        sb = _make_sandbox()
        result = sb.reduce_position("BTC", 0.1)
        assert result["status"] == "error"

    def test_perp_summary(self):
        sb = _make_sandbox()
        sb.set_leverage("BTC", 3)
        price = sb.get_price("BTC")
        sb.long("BTC", 0.01, price, order_type="market")
        summary = sb.get_perp_summary()
        assert "account_value" in summary
        assert "positions" in summary
        assert len(summary["positions"]) == 1
        assert summary["positions"][0]["asset"] == "BTC"

    def test_get_perp_position_detail(self):
        sb = _make_sandbox()
        sb.set_leverage("BTC", 3)
        price = sb.get_price("BTC")
        sb.long("BTC", 0.01, price, order_type="market")
        pos = sb.get_perp_position("BTC")
        assert pos is not None
        assert pos["asset"] == "BTC"
        assert pos["size"] > 0
        assert "entry_price" in pos

    def test_get_perp_position_none_when_flat(self):
        sb = _make_sandbox()
        pos = sb.get_perp_position("BTC")
        assert pos is None


class TestSpotSummary:
    def test_spot_summary_balances(self):
        sb = _make_sandbox()
        summary = sb.get_spot_summary()
        assert "balances" in summary
        usdc_bal = next(b for b in summary["balances"] if b["asset"] == "USDC")
        assert usdc_bal["total"] == 10000.0


class TestOrderManagement:
    def test_limit_order_pending(self):
        sb = _make_sandbox()
        sb.set_leverage("BTC", 3)
        # Place a limit order far below market
        result = sb.long("BTC", 0.01, 1000.0, order_type="limit")
        assert result["status"] == "success"
        orders = sb.get_perp_open_orders("BTC")
        assert len(orders) == 1

    def test_cancel_order(self):
        sb = _make_sandbox()
        sb.set_leverage("BTC", 3)
        sb.long("BTC", 0.01, 1000.0, order_type="limit")
        orders = sb.get_perp_open_orders("BTC")
        oid = orders[0]["id"]
        result = sb.cancel_perp_orders("BTC", [oid])
        assert result["status"] == "success"
        assert len(sb.get_perp_open_orders("BTC")) == 0


class TestTimeAdvancement:
    def test_advance_time(self):
        sb = _make_sandbox()
        start = sb.current_time
        new_time = sb.advance_time("1h")
        assert new_time is not None
        assert new_time > start

    def test_advance_time_past_end(self):
        sb = _make_sandbox()
        # Advance way past end date
        for _ in range(200):
            result = sb.advance_time("1h")
            if result is None:
                break
        # Should eventually return None
        assert sb.current_time <= datetime(2025, 1, 6, tzinfo=timezone.utc)

    def test_get_current_time(self):
        sb = _make_sandbox()
        assert sb.get_current_time() == sb.current_time


class TestFees:
    def test_trading_fees_tracked(self):
        sb = _make_sandbox()
        sb.set_leverage("BTC", 3)
        price = sb.get_price("BTC")
        sb.long("BTC", 0.1, price, order_type="market")
        assert sb.total_tx_fees > 0
