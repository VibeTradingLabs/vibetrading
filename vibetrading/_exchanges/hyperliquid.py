"""
Hyperliquid exchange sandbox for live trading.

Requires: pip install vibetrading[hyperliquid]
  (hyperliquid-python-sdk, eth_account)
"""

import os
import math
import logging
import traceback
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np

from .base import LiveSandboxBase
from .._core.sandbox_base import SUPPORTED_INTERVALS, SUPPORTED_LEVERAGE
from .._models.orders import (
    PerpAccountSummary, PerpPositionSummary,
    SpotAccountSummary, SpotBalanceSummary,
    SpotOrder, PerpOrder,
    SpotOrderResponse, PerpOrderResponse,
    CancelOrdersResponse,
)
from .._models.types import SpotMeta, PerpMeta
from .._utils.math import truncate_quantity, format_hyperliquid_price
from .._utils.notification import NotificationDeduplicator

logger = logging.getLogger(__name__)

try:
    import eth_account
    from hyperliquid.exchange import Exchange
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    _HAS_HL = True
except ImportError:
    _HAS_HL = False


class HyperliquidSandbox(LiveSandboxBase):
    """Live trading sandbox for Hyperliquid L1 (perps + spot)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        mode: str = "live",
        notification_deduplicator: Optional[NotificationDeduplicator] = None,
        builder_address: Optional[str] = None,
        **kwargs,
    ):
        if not _HAS_HL:
            raise ImportError(
                "hyperliquid SDK not installed. Install with: "
                "pip install vibetrading[hyperliquid]"
            )

        super().__init__(
            exchange_name="hyperliquid",
            api_key=api_key,
            api_secret=api_secret,
            mode=mode,
            notification_deduplicator=notification_deduplicator,
            **kwargs,
        )
        self.builder_address = builder_address

        account = eth_account.Account.from_key(api_secret)
        self.hyperliquid = Exchange(wallet=account, account_address=api_key)
        self.info: Info = self.hyperliquid.info

        self._load_market_metadata()
        logger.info("HyperliquidSandbox ready (mode=%s)", mode)

    # ── Market metadata ────────────────────────────────────────────────
    def _load_market_metadata(self):
        try:
            meta = self.info.meta()
            for asset in meta.get("universe", []):
                name = asset["name"]
                sz_dec = int(asset.get("szDecimals", 6))
                max_lev = int(asset.get("maxLeverage", 20))
                self.perp_meta[name] = PerpMeta(
                    symbol=f"{name}/USDC:USDC", name=name,
                    sz_decimals=sz_dec, price_decimals=6 - sz_dec,
                    max_leverage=max_lev,
                )
            spot_info = self.info.spot_meta()
            tokens = spot_info["tokens"]
            for m in spot_info["universe"]:
                if len(m["tokens"]) == 2 and m["tokens"][1] == 0:
                    idx = m["tokens"][0]
                    tok = tokens[idx]
                    tok_name = tok["name"]
                    full_name = tok.get("fullName")
                    if full_name and full_name.startswith("Unit") and tok_name.startswith("U"):
                        tok_name = tok_name[1:]
                    sz_dec = int(tok["szDecimals"])
                    self.spot_meta[tok_name] = SpotMeta(
                        symbol=f"{tok_name}/USDC", name=tok_name,
                        sz_decimals=sz_dec, price_decimals=8 - sz_dec,
                    )
                    self.asset_spot_mapping[tok_name] = m["name"]
                    self.spot_asset_mapping[m["name"]] = tok_name
            self.supported_assets = list(self.perp_meta.keys())
        except Exception as e:
            logger.error("Failed to load Hyperliquid market metadata: %s", e)

    # ── VibeSandboxBase implementation (delegating to hyperliquid SDK) ──
    def get_supported_assets(self) -> List[str]:
        return self.supported_assets

    def get_price(self, asset: str) -> float:
        try:
            all_mids = self.info.all_mids()
            return float(all_mids.get(asset, float("nan")))
        except Exception:
            return float("nan")

    def get_spot_price(self, asset: str) -> float:
        return self.get_price(asset)

    def get_perp_price(self, asset: str) -> float:
        return self.get_price(asset)

    def my_spot_balance(self, asset: str) -> float:
        try:
            state = self.info.spot_user_state(self.api_key)
            for b in state.get("balances", []):
                coin = b.get("coin", "")
                mapped = self.spot_asset_mapping.get(coin, coin)
                if mapped == asset:
                    return float(b.get("total", 0.0))
        except Exception as e:
            logger.error("Error getting spot balance: %s", e)
        return 0.0

    def my_futures_balance(self, asset: str = "USDC") -> float:
        try:
            state = self.info.user_state(self.api_key)
            if asset == "USDC":
                mv = state.get("marginSummary", {})
                return float(mv.get("accountValue", 0.0))
        except Exception as e:
            logger.error("Error getting futures balance: %s", e)
        return 0.0

    def buy(self, asset: str, quantity: float, price: float, order_type: str = "limit") -> Dict[str, Any]:
        spot_name = self.asset_spot_mapping.get(asset, asset)
        pm = self.spot_meta.get(asset)
        if pm:
            quantity = truncate_quantity(quantity, pm.sz_decimals)
            price = format_hyperliquid_price(price, is_spot=True)
        is_market = order_type == "market"
        resp = self.hyperliquid.order(spot_name, True, quantity, price, {"limit": {"tif": "Ioc" if is_market else "Gtc"}})
        return self._parse_order_response(resp, asset, "buy", quantity, price, "spot")

    def sell(self, asset: str, quantity: float, price: float, order_type: str = "limit") -> Dict[str, Any]:
        spot_name = self.asset_spot_mapping.get(asset, asset)
        pm = self.spot_meta.get(asset)
        if pm:
            quantity = truncate_quantity(quantity, pm.sz_decimals)
            price = format_hyperliquid_price(price, is_spot=True)
        is_market = order_type == "market"
        resp = self.hyperliquid.order(spot_name, False, quantity, price, {"limit": {"tif": "Ioc" if is_market else "Gtc"}})
        return self._parse_order_response(resp, asset, "sell", quantity, price, "spot")

    def long(self, asset: str, quantity: float, price: float, order_type: str = "limit") -> Dict[str, Any]:
        pm = self.perp_meta.get(asset)
        if pm:
            quantity = truncate_quantity(quantity, pm.sz_decimals)
            price = format_hyperliquid_price(price, is_spot=False)
        is_market = order_type == "market"
        resp = self.hyperliquid.order(asset, True, quantity, price, {"limit": {"tif": "Ioc" if is_market else "Gtc"}})
        return self._parse_order_response(resp, asset, "long", quantity, price, "futures")

    def short(self, asset: str, quantity: float, price: float, order_type: str = "limit") -> Dict[str, Any]:
        pm = self.perp_meta.get(asset)
        if pm:
            quantity = truncate_quantity(quantity, pm.sz_decimals)
            price = format_hyperliquid_price(price, is_spot=False)
        is_market = order_type == "market"
        resp = self.hyperliquid.order(asset, False, quantity, price, {"limit": {"tif": "Ioc" if is_market else "Gtc"}})
        return self._parse_order_response(resp, asset, "short", quantity, price, "futures")

    def reduce_position(self, asset: str, quantity: float) -> Dict[str, Any]:
        pos = self.get_futures_position(asset)
        if pos == 0:
            return PerpOrderResponse.Error(f"No position for {asset}").to_dict()
        if pos > 0:
            return self.short(asset, min(quantity, abs(pos)), self.get_price(asset), "market")
        else:
            return self.long(asset, min(quantity, abs(pos)), self.get_price(asset), "market")

    def set_leverage(self, asset: str, leverage: int):
        try:
            self.hyperliquid.update_leverage(leverage, asset, is_cross=True)
        except Exception as e:
            logger.error("Error setting leverage: %s", e)

    def get_futures_position(self, asset: str) -> float:
        try:
            state = self.info.user_state(self.api_key)
            for p in state.get("assetPositions", []):
                pos = p.get("position", {})
                if pos.get("coin") == asset:
                    return float(pos.get("szi", 0.0))
        except Exception as e:
            logger.error("Error getting position: %s", e)
        return 0.0

    def get_spot_ohlcv(self, asset: str, interval: str, limit: int) -> pd.DataFrame:
        return self._fetch_ohlcv(asset, interval, limit, is_spot=True)

    def get_futures_ohlcv(self, asset: str, interval: str, limit: int) -> pd.DataFrame:
        return self._fetch_ohlcv(asset, interval, limit, is_spot=False)

    def _fetch_ohlcv(self, asset: str, interval: str, limit: int, is_spot: bool = False) -> pd.DataFrame:
        try:
            now_ms = int(time.time() * 1000)
            interval_ms = SUPPORTED_INTERVALS.get(interval, 3600) * 1000
            start_ms = now_ms - interval_ms * limit
            candles = self.info.candles_snapshot(asset, interval, start_ms, now_ms)
            if not candles:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            rows = []
            for c in candles:
                rows.append({
                    "timestamp": pd.to_datetime(c["t"], unit="ms", utc=True),
                    "open": float(c["o"]), "high": float(c["h"]),
                    "low": float(c["l"]), "close": float(c["c"]),
                    "volume": float(c["v"]),
                })
            df = pd.DataFrame(rows).set_index("timestamp").sort_index().tail(limit)
            return df
        except Exception as e:
            logger.error("OHLCV fetch error: %s", e)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    def get_funding_rate(self, asset: str = "BTC", timestamp=None) -> float:
        try:
            meta = self.info.meta()
            for u in meta.get("universe", []):
                if u["name"] == asset:
                    return float(u.get("funding", 0.0))
        except Exception:
            pass
        return 0.0

    def get_funding_rate_history(self, asset: str, limit: int) -> pd.DataFrame:
        try:
            now_ms = int(time.time() * 1000)
            start_ms = now_ms - limit * 3600 * 1000
            data = self.info.funding_history(asset, start_ms, end_time=now_ms)
            if data:
                rows = [{"timestamp": pd.to_datetime(d["time"], unit="ms", utc=True),
                         "fundingRate": float(d.get("fundingRate", 0))} for d in data]
                return pd.DataFrame(rows)
        except Exception:
            pass
        return pd.DataFrame(columns=["timestamp", "fundingRate"])

    def get_open_interest(self, asset: str = "BTC", timestamp=None) -> float:
        try:
            meta = self.info.meta()
            for u in meta.get("universe", []):
                if u.get("name") == asset:
                    return float(u.get("openInterest", 0.0))
        except Exception:
            pass
        return 0.0

    def get_open_interest_history(self, asset: str, limit: int) -> pd.DataFrame:
        return pd.DataFrame(columns=["timestamp", "openInterest"])

    def cancel_order(self, order_id: str) -> bool:
        try:
            resp = self.hyperliquid.cancel(order_id, None)
            return resp.get("status") == "ok" if isinstance(resp, dict) else False
        except Exception as e:
            logger.error("Cancel order error: %s", e)
            return False

    def get_perp_open_orders(self, asset: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            orders = self.info.open_orders(self.api_key)
            result = []
            for o in orders:
                if asset and o.get("coin") != asset:
                    continue
                result.append({
                    "order_id": str(o.get("oid", "")),
                    "asset": o.get("coin", ""),
                    "side": o.get("side", ""),
                    "price": float(o.get("limitPx", 0)),
                    "amount": float(o.get("sz", 0)),
                    "symbol": o.get("coin", ""),
                })
            return result
        except Exception as e:
            logger.error("Open orders error: %s", e)
            return []

    def get_spot_open_orders(self, asset: Optional[str] = None) -> List[Dict[str, Any]]:
        return []

    def cancel_spot_orders(self, asset: str, order_ids: List[str]) -> Dict[str, Any]:
        return CancelOrdersResponse(status="success", orders=[]).to_dict()

    def cancel_perp_orders(self, asset: str, order_ids: List[str]) -> Dict[str, Any]:
        results = []
        for oid in order_ids:
            if self.cancel_order(oid):
                results.append({"status": "success", "id": oid})
            else:
                results.append({"status": "error", "id": oid})
        return CancelOrdersResponse(status="success", orders=results).to_dict()

    def get_perp_summary(self) -> Dict[str, Any]:
        try:
            state = self.info.user_state(self.api_key)
            ms = state.get("marginSummary", {})
            positions = []
            for p in state.get("assetPositions", []):
                pos = p.get("position", {})
                sz = float(pos.get("szi", 0))
                if abs(sz) < 1e-8:
                    continue
                positions.append(PerpPositionSummary(
                    asset=pos.get("coin", ""),
                    size=sz,
                    entry_price=float(pos.get("entryPx", 0)),
                    unrealized_pnl=float(pos.get("unrealizedPnl", 0)),
                    position_value=float(pos.get("positionValue", 0)),
                    margin_used=float(pos.get("marginUsed", 0)),
                ))
            summary = PerpAccountSummary(
                account_value=float(ms.get("accountValue", 0)),
                available_margin=float(ms.get("totalMarginUsed", 0)),
                total_margin_used=float(ms.get("totalMarginUsed", 0)),
                total_unrealized_pnl=float(ms.get("totalNtlPos", 0)),
                positions=positions,
            )
            return summary.to_dict()
        except Exception as e:
            logger.error("Perp summary error: %s", e)
            return PerpAccountSummary().to_dict()

    def get_perp_position(self, asset: str) -> Optional[Dict[str, Any]]:
        for p in self.get_perp_summary().get("positions", []):
            if p.get("asset") == asset:
                return p
        return None

    def get_spot_summary(self) -> Dict[str, Any]:
        try:
            state = self.info.spot_user_state(self.api_key)
            bals = []
            for b in state.get("balances", []):
                coin = b.get("coin", "")
                mapped = self.spot_asset_mapping.get(coin, coin)
                bals.append(SpotBalanceSummary(
                    asset=mapped, total=float(b.get("total", 0)),
                    free=float(b.get("total", 0)), locked=0,
                ))
            return SpotAccountSummary(balances=bals).to_dict()
        except Exception as e:
            logger.error("Spot summary error: %s", e)
            return SpotAccountSummary().to_dict()

    def get_futures_unrealized_pnl(self, asset=None):
        return 0.0

    def get_total_futures_unrealized_pnl(self) -> float:
        return 0.0

    def get_available_margin(self) -> float:
        return self.my_futures_balance("USDC")

    def get_total_margin_used(self) -> float:
        return 0.0

    def apply_funding_payment(self, asset, position_size, leverage=1):
        pass

    def advance_time(self, interval):
        return None

    # ── Helpers ─────────────────────────────────────────────────────────
    def _parse_order_response(self, resp: Any, asset: str, side: str,
                              qty: float, price: float, trading_type: str) -> Dict[str, Any]:
        try:
            status = resp.get("status", "error") if isinstance(resp, dict) else "error"
            resting = None
            if isinstance(resp, dict) and "response" in resp:
                inner = resp["response"]
                if isinstance(inner, dict) and "data" in inner:
                    data = inner["data"]
                    if isinstance(data, dict) and "statuses" in data:
                        for s in data["statuses"]:
                            if "resting" in s:
                                resting = s["resting"]
                            elif "filled" in s:
                                resting = s["filled"]
            oid = str(resting.get("oid", "")) if isinstance(resting, dict) else ""
            ts = int(time.time())
            if trading_type == "spot":
                order = SpotOrder(id=oid, asset=asset, side=side, type="limit", size=qty, price=price, timestamp=ts)
                return SpotOrderResponse(status="success", order=order.to_dict()).to_dict()
            else:
                order = PerpOrder(id=oid, asset=asset, side=side, type="limit", size=qty, price=price, timestamp=ts)
                return PerpOrderResponse(status="success", order=order.to_dict()).to_dict()
        except Exception as e:
            logger.error("Parse order response error: %s", e)
            if trading_type == "spot":
                return SpotOrderResponse.Error(str(e)).to_dict()
            return PerpOrderResponse.Error(str(e)).to_dict()
