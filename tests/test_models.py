"""Tests for order and account models."""

from vibetrading._models.orders import (
    CancelOrdersResponse,
    PerpAccountSummary,
    PerpOrder,
    PerpOrderResponse,
    PerpPositionSummary,
    SpotAccountSummary,
    SpotBalanceSummary,
    SpotOrderResponse,
)
from vibetrading._models.types import AgentMetadata, PerpMeta, RiskLevel, SpotMeta, StrategyType


class TestPerpModels:
    def test_position_summary_to_dict(self):
        pos = PerpPositionSummary(
            asset="BTC",
            size=0.1,
            entry_price=68000.0,
            unrealized_pnl=125.50,
            position_value=6800.0,
            margin_used=1360.0,
        )
        d = pos.to_dict()
        assert d["asset"] == "BTC"
        assert d["size"] == 0.1
        assert d["entry_price"] == 68000.0

    def test_account_summary_with_positions(self):
        pos = PerpPositionSummary(
            asset="ETH",
            size=-1.0,
            entry_price=3500.0,
            unrealized_pnl=-50.0,
            position_value=3500.0,
            margin_used=700.0,
        )
        summary = PerpAccountSummary(
            account_value=10000.0,
            available_margin=9300.0,
            total_margin_used=700.0,
            total_unrealized_pnl=-50.0,
            positions=[pos],
        )
        d = summary.to_dict()
        assert d["account_value"] == 10000.0
        assert len(d["positions"]) == 1
        assert d["positions"][0]["asset"] == "ETH"

    def test_perp_order_response_success(self):
        order = PerpOrder(
            id="1",
            asset="BTC",
            side="long",
            type="limit",
            size=0.1,
            price=68000.0,
            timestamp=1700000000,
        )
        resp = PerpOrderResponse(status="success", order=order)
        d = resp.to_dict()
        assert d["status"] == "success"
        assert d["order"]["id"] == "1"

    def test_perp_order_response_error(self):
        resp = PerpOrderResponse.Error("Insufficient margin")
        d = resp.to_dict()
        assert d["status"] == "error"
        assert d["error"] == "Insufficient margin"
        assert d["order"] is None


class TestSpotModels:
    def test_spot_balance_summary(self):
        bal = SpotBalanceSummary(asset="USDC", total=10000.0, free=8000.0, locked=2000.0)
        d = bal.to_dict()
        assert d["asset"] == "USDC"
        assert d["free"] == 8000.0

    def test_spot_account_summary(self):
        bals = [
            SpotBalanceSummary(asset="USDC", total=10000.0, free=10000.0, locked=0),
            SpotBalanceSummary(asset="BTC", total=0.5, free=0.5, locked=0),
        ]
        summary = SpotAccountSummary(balances=bals)
        d = summary.to_dict()
        assert len(d["balances"]) == 2

    def test_spot_order_response_error(self):
        resp = SpotOrderResponse.Error("Cannot buy USDC")
        assert resp.status == "error"
        assert resp.error == "Cannot buy USDC"


class TestCancelOrdersResponse:
    def test_cancel_success(self):
        resp = CancelOrdersResponse(
            status="success",
            orders=[{"status": "success", "id": "1"}, {"status": "success", "id": "2"}],
        )
        d = resp.to_dict()
        assert len(d["orders"]) == 2

    def test_cancel_error(self):
        resp = CancelOrdersResponse.Error("Not found")
        assert resp.status == "error"


class TestTypeModels:
    def test_spot_meta(self):
        meta = SpotMeta(asset="BTC", symbol="BTC/USDT", sz_decimals=5, price_precision=2)
        assert meta.asset == "BTC"

    def test_perp_meta(self):
        meta = PerpMeta(asset="ETH", symbol="ETH/USDT:USDT", sz_decimals=4, price_precision=2, max_leverage=20)
        assert meta.max_leverage == 20

    def test_agent_metadata_defaults(self):
        meta = AgentMetadata()
        assert meta.strategy_type == StrategyType.CUSTOM
        assert meta.risk_level == RiskLevel.MEDIUM
        assert meta.exchange == "hyperliquid"
