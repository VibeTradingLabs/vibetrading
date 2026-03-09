import time

import pandas as pd
import pytest

from vibetrading._exchanges.hyperliquid import HyperliquidSandbox
from vibetrading._models.types import PerpMeta, SpotMeta


class FakeInfo:
    def __init__(self):
        self._mids = {"BTC": "100.0"}

    def all_mids(self):
        return self._mids

    def spot_user_state(self, api_key):
        assert api_key == "acct"
        return {
            "balances": [
                {"coin": "USDC", "total": "10"},
                {"coin": "UBTC", "total": "1.5"},
            ]
        }

    def user_state(self, api_key):
        assert api_key == "acct"
        return {
            "marginSummary": {"accountValue": "123.45", "totalMarginUsed": "7"},
            "assetPositions": [
                {
                    "position": {
                        "coin": "BTC",
                        "szi": "2",
                        "entryPx": "99",
                        "unrealizedPnl": "1",
                        "positionValue": "200",
                        "marginUsed": "5",
                    }
                },
                {"position": {"coin": "ETH", "szi": "0"}},
            ],
        }

    def candles_snapshot(self, asset, interval, start_ms, now_ms):
        assert asset == "BTC"
        assert interval in {"1h", "1m", "5m", "15m"}
        assert start_ms < now_ms
        return [
            {"t": now_ms - 2000, "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "10"},
            {"t": now_ms - 1000, "o": "1.5", "h": "2.5", "l": "1", "c": "2", "v": "12"},
        ]

    def open_orders(self, api_key):
        assert api_key == "acct"
        return [
            {"oid": 1, "coin": "BTC", "side": "B", "limitPx": "100", "sz": "0.1"},
            {"oid": 2, "coin": "ETH", "side": "S", "limitPx": "200", "sz": "0.2"},
        ]

    def funding_history(self, asset, start_ms, end_time):
        assert asset == "BTC"
        return [{"time": end_time - 1000, "fundingRate": "0.001"}]

    def meta(self):
        return {"universe": [{"name": "BTC", "funding": "0.01", "openInterest": "99"}]}


class FakeExchange:
    def __init__(self):
        self.calls = []

    def order(self, coin, is_buy, sz, px, opts):
        self.calls.append((coin, is_buy, sz, px, opts))
        # Match the shape HyperliquidSandbox._parse_order_response expects.
        return {
            "status": "ok",
            "response": {"data": {"statuses": [{"resting": {"oid": 123}}]}},
        }

    def cancel(self, order_id, _):
        self.calls.append(("cancel", order_id))
        return {"status": "ok"}

    def update_leverage(self, leverage, asset, is_cross=True):
        self.calls.append(("update_leverage", leverage, asset, is_cross))


@pytest.fixture()
def sandbox():
    sb = HyperliquidSandbox.__new__(HyperliquidSandbox)

    sb.api_key = "acct"
    sb.api_secret = "secret"

    sb.hyperliquid = FakeExchange()
    sb.info = FakeInfo()

    sb.perp_meta = {"BTC": PerpMeta(asset="BTC", symbol="BTC/USDC:USDC", sz_decimals=3, price_precision=3, max_leverage=20)}
    sb.spot_meta = {"BTC": SpotMeta(asset="BTC", symbol="BTC/USDC", sz_decimals=3, price_precision=3)}

    # Hyperliquid spot names differ; map asset->spot book name
    sb.asset_spot_mapping = {"BTC": "UBTC/USDC"}
    sb.spot_asset_mapping = {"UBTC": "BTC"}

    sb.supported_assets = ["BTC"]

    return sb


def test_prices_and_balances(sandbox):
    assert sandbox.get_price("BTC") == 100.0
    assert sandbox.get_spot_price("BTC") == 100.0
    assert sandbox.get_perp_price("BTC") == 100.0

    assert sandbox.my_spot_balance("BTC") == 1.5
    assert sandbox.my_futures_balance("USDC") == 123.45


def test_fetch_ohlcv(sandbox):
    df = sandbox.get_spot_ohlcv("BTC", "1h", limit=2)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df.index.tz is not None


def test_open_orders_and_cancel(sandbox):
    orders = sandbox.get_perp_open_orders(asset="BTC")
    assert len(orders) == 1
    assert orders[0]["order_id"] == "1"

    assert sandbox.cancel_order("1") is True


def test_buy_sell_long_short_and_parse_response(sandbox, monkeypatch):
    # Freeze time for deterministic timestamps.
    monkeypatch.setattr(time, "time", lambda: 1700000000)

    buy = sandbox.buy("BTC", quantity=1.23456, price=12345.0, order_type="limit")
    assert buy["status"] == "success"
    assert buy["order"]["id"] == "123"

    sell = sandbox.sell("BTC", quantity=1.23456, price=12345.0, order_type="market")
    assert sell["status"] == "success"

    long = sandbox.long("BTC", quantity=1.23456, price=123.456789, order_type="limit")
    assert long["status"] == "success"

    short = sandbox.short("BTC", quantity=1.23456, price=123.456789, order_type="market")
    assert short["status"] == "success"

    # Verify the underlying SDK was called with formatted values.
    # Quantity should be truncated to 3 decimals.
    called = sandbox.hyperliquid.calls
    assert called[0][0] == "UBTC/USDC"  # spot mapping
    assert called[0][2] == 1.234


def test_summaries_and_funding(sandbox):
    perp = sandbox.get_perp_summary()
    assert perp["account_value"] == 123.45
    assert len(perp["positions"]) == 1
    assert perp["positions"][0]["asset"] == "BTC"

    # get_perp_position is a thin convenience wrapper
    pos = sandbox.get_perp_position("BTC")
    assert pos is not None
    assert pos["asset"] == "BTC"

    spot = sandbox.get_spot_summary()
    assert any(b["asset"] == "BTC" for b in spot["balances"])

    assert sandbox.get_funding_rate("BTC") == 0.01
    fh = sandbox.get_funding_rate_history("BTC", limit=1)
    assert not fh.empty


def test_reduce_position_and_cancel_perp_orders(sandbox):
    # With a positive position, reduce_position should route through short(..., market)
    resp = sandbox.reduce_position("BTC", quantity=0.5)
    assert resp["status"] == "success"

    cancel = sandbox.cancel_perp_orders("BTC", ["1", "2"])
    assert cancel["status"] == "success"
    assert len(cancel["orders"]) == 2
