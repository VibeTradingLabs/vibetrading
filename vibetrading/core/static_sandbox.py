"""
StaticSandbox - Backtesting sandbox using pre-downloaded historical data.

Provides a complete trading simulation environment that uses historical data
to simulate spot and futures trading. Data must be downloaded in advance
(e.g. via ``vibetrading.tools.data_downloader``) or passed directly as
DataFrames.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
from datetime import datetime, timedelta, timezone
import numpy as np
import os
import math
import logging

from .sandbox_base import VibeSandboxBase, SUPPORTED_LEVERAGE
from ..config import DATASET_DIR
from ..tools.data_loader import load_csv, generate_cache_filename
from ..models.orders import (
    PerpAccountSummary,
    PerpPositionSummary,
    SpotAccountSummary,
    SpotBalanceSummary,
    SpotOrder,
    PerpOrder,
    SpotOrderResponse,
    PerpOrderResponse,
    CancelOrdersResponse,
)
from ..utils.logging import (
    log_trade_execution,
    configure_rate_limiting,
    log_download_success,
    log_download_error,
)

ENABLE_STRUCTURED_LOGGING = True
ENABLE_VERBOSE_LOGS = False

logger = logging.getLogger(__name__)


# Default market metadata (no network required)
DEFAULT_ASSETS = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX",
    "LINK", "DOT", "SUI", "ARB", "OP", "PEPE", "WIF",
]

DEFAULT_PERP_MARKETS = {
    a: {
        "symbol": f"{a}/USDT:USDT", "name": a,
        "sz_decimals": 4, "price_decimals": 2, "max_leverage": 20,
    }
    for a in DEFAULT_ASSETS
}
# Override precision for high-value assets
DEFAULT_PERP_MARKETS["BTC"]["sz_decimals"] = 5
DEFAULT_PERP_MARKETS["BTC"]["price_decimals"] = 1
DEFAULT_PERP_MARKETS["ETH"]["sz_decimals"] = 4
DEFAULT_PERP_MARKETS["ETH"]["price_decimals"] = 2

DEFAULT_SPOT_MARKETS = {
    a: {
        "name": a, "symbol": f"{a}/USDT",
        "sz_decimals": 4, "price_decimals": 4,
    }
    for a in DEFAULT_ASSETS
}


class StaticSandbox(VibeSandboxBase):
    """Backtesting sandbox using pre-downloaded data from CSV cache.

    Data is loaded from CSV files in the dataset directory. If data is
    not found, a clear error is raised directing the user to download
    data first.

    Data can also be passed directly via the ``data`` parameter as a dict
    mapping ``(symbol, interval)`` tuples to DataFrames.
    """

    INITIAL_USDC_BALANCE = 10000.0
    DEFAULT_FEE_RATE = 0.001
    DATASET_FOLDER = DATASET_DIR
    STABLECOINS: set = {"USDT", "USDC", "USDE", "DAI", "TUSD"}

    def __init__(
        self,
        exchange: str = "binance",
        start_date: str = "2025-01-01",
        end_date: str = "2025-07-01",
        initial_balances: Optional[Dict[str, float]] = None,
        fee_rate: Optional[float] = None,
        supported_assets: Optional[List[str]] = None,
        mute_strategy_prints: bool = False,
        data: Optional[Dict[Tuple[str, str], pd.DataFrame]] = None,
    ):
        """
        Args:
            exchange: Exchange name (used for CSV cache file lookup).
            start_date: Backtest start date (YYYY-MM-DD).
            end_date: Backtest end date (YYYY-MM-DD).
            initial_balances: Starting asset balances (default: {"USDC": 10000}).
            fee_rate: Trading fee rate (default: 0.001).
            supported_assets: List of supported asset names.
            mute_strategy_prints: Suppress print output from strategy code.
            data: Pre-loaded data dict mapping (symbol, interval) -> DataFrame.
                  If provided, CSV cache is skipped entirely.
        """
        import sys
        self.exchange = exchange.lower()
        self.start_date = start_date
        self.end_date = end_date
        self.mute_strategy_prints = mute_strategy_prints

        configure_rate_limiting(max_qps=1000.0, window_seconds=1.0)

        self.config: Dict[str, Any] = {"fee_rate": self.DEFAULT_FEE_RATE}
        self.fee_rate = fee_rate if fee_rate is not None else self.DEFAULT_FEE_RATE

        # Balances & tracking
        self.balances: Dict[str, float] = initial_balances.copy() if initial_balances else {"USDC": self.INITIAL_USDC_BALANCE}
        self.trades: List[Dict[str, Any]] = []
        self.futures_positions: Dict[str, float] = {}
        self.futures_position_details: Dict[str, Dict[str, Any]] = {}
        self.locked_margin: float = 0.0
        self.accumulated_funding_fee: float = 0.0
        self.funding_payments: List[Dict[str, Any]] = []
        self.position_tracking: Dict[str, Dict[str, Any]] = {}
        self.total_tx_fees: float = 0.0

        # Order management
        self.pending_orders: Dict[str, Dict] = {}
        self.filled_orders: List[Dict] = []
        self.cancelled_orders: List[Dict] = []
        self._next_order_id: int = 1

        # Data cache (loaded from CSV or passed directly)
        self.data_cache: Dict[Tuple[str, str], pd.DataFrame] = data.copy() if data else {}
        self._price_cache: Dict[Tuple[str, datetime], Tuple[float, datetime]] = {}

        # Time
        start_dt = datetime.strptime(self.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        self.current_time: datetime = start_dt

        # Markets (static, no network required)
        if supported_assets is not None:
            self.supported_assets = [a.upper() for a in supported_assets]
            self.perp_markets: Dict[str, Any] = {
                a: DEFAULT_PERP_MARKETS.get(a, {
                    "symbol": f"{a}/USDT:USDT", "name": a,
                    "sz_decimals": 4, "price_decimals": 2, "max_leverage": 20,
                })
                for a in self.supported_assets
            }
            self.spot_markets: Dict[str, Any] = {
                a: DEFAULT_SPOT_MARKETS.get(a, {
                    "name": a, "symbol": f"{a}/USDT",
                    "sz_decimals": 4, "price_decimals": 4,
                })
                for a in self.supported_assets
            }
        else:
            self.supported_assets = list(DEFAULT_ASSETS)
            self.perp_markets = dict(DEFAULT_PERP_MARKETS)
            self.spot_markets = dict(DEFAULT_SPOT_MARKETS)

        print(f"StaticSandbox ready: {self.exchange} | {self.start_date} -> {self.end_date}")
        sys.stdout.flush()

    # ── Symbol helpers ─────────────────────────────────────────────────
    def _normalize_asset(self, asset: str) -> str:
        return asset.upper()

    def _asset_to_spot_symbol(self, asset: str) -> str:
        m = self.spot_markets.get(asset)
        if m:
            return m["symbol"]
        raise ValueError(f"Asset {asset} not found in spot markets")

    def _asset_to_futures_symbol(self, asset: str) -> str:
        m = self.perp_markets.get(asset)
        if m:
            return m["symbol"]
        raise ValueError(f"Asset {asset} not found in perp markets")

    def get_supported_assets(self) -> List[str]:
        return list(self.supported_assets)

    # ── Data loading (CSV cache only, no network) ─────────────────────
    def _is_futures_symbol(self, symbol: str) -> bool:
        return ":" in symbol

    def _load_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load data from in-memory cache or CSV files. No network access."""
        cache_key = (symbol, timeframe)
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]

        # Try CSV cache file (written by download_data())
        cache_file = generate_cache_filename(
            self.exchange, symbol, self.start_date, self.end_date,
            timeframe, self.DATASET_FOLDER,
        )
        if os.path.exists(cache_file):
            try:
                data = load_csv(cache_file)
                if not data.empty:
                    self.data_cache[cache_key] = data
                    log_download_success(
                        f"Loaded {len(data)} cached records for {symbol} {timeframe}",
                        items_count=len(data),
                    )
                    return data
            except Exception as e:
                log_download_error(f"Cache load failed: {e}", error_type="cache")

        # No data found - give a clear error message
        raise RuntimeError(
            f"No data found for {symbol} ({timeframe}). "
            f"Please download data first:\n\n"
            f"  from vibetrading.tools import download_data\n"
            f"  download_data(['{symbol.split('/')[0]}'], exchange='{self.exchange}', "
            f"interval='{timeframe}', ...)\n"
        )

    # ── Validation ─────────────────────────────────────────────────────
    def _validate_asset(self, asset: str) -> bool:
        if asset not in self.supported_assets:
            if ENABLE_VERBOSE_LOGS:
                print(f"Asset '{asset}' not supported on {self.exchange}")
            return False
        return True

    def _validate_timeframe(self, tf: str):
        if tf not in ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]:
            raise ValueError(f"Unsupported timeframe: {tf}")

    def _validate_limit(self, limit: int):
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError(f"Limit must be a positive integer, got {limit}")

    def _validate_leverage(self, leverage: int):
        if leverage not in SUPPORTED_LEVERAGE:
            raise ValueError(f"Leverage must be one of {SUPPORTED_LEVERAGE}, got {leverage}")

    # ── Balance ────────────────────────────────────────────────────────
    def my_spot_balance(self, asset: str) -> float:
        return self.balances.get(asset, 0.0)

    def my_futures_balance(self, asset: str) -> float:
        return self.balances.get(asset, 0.0)

    # ── Price ──────────────────────────────────────────────────────────
    def get_price(self, asset: str) -> float:
        asset = self._normalize_asset(asset)
        if asset.upper() in self.STABLECOINS:
            return 1.0
        if not self._validate_asset(asset):
            return float("nan")
        try:
            return self._get_futures_price(asset)
        except RuntimeError:
            try:
                return self._get_spot_price(asset)
            except RuntimeError:
                return float("nan")

    def get_spot_price(self, asset: str) -> float:
        return self.get_price(asset)

    def get_perp_price(self, asset: str) -> float:
        return self.get_price(asset)

    def _get_optimal_timeframe(self) -> str:
        if hasattr(self, "_backtest_interval"):
            return self._backtest_interval
        return "1h"

    def _get_spot_price(self, asset: str) -> float:
        ck = (f"spot_{asset}", self.current_time)
        cached = self._get_cached_price(ck)
        if cached is not None:
            return cached
        symbol = self._asset_to_spot_symbol(asset)
        data = self._load_data(symbol, self._get_optimal_timeframe())
        if data.empty:
            raise RuntimeError(f"No spot data for {asset}")
        t = data.index.asof(self.current_time)
        if pd.isna(t):
            t = data.index[0]
        price = float(np.asarray(data.loc[t, "close"]).item())
        self._cache_price(ck, price)
        return price

    def _get_futures_price(self, asset: str) -> float:
        ck = (f"futures_{asset}", self.current_time)
        cached = self._get_cached_price(ck)
        if cached is not None:
            return cached
        symbol = self._asset_to_futures_symbol(asset)
        data = self._load_data(symbol, self._get_optimal_timeframe())
        if data.empty:
            raise RuntimeError(f"No futures data for {asset}")
        t = data.index.asof(self.current_time)
        if pd.isna(t):
            t = data.index[0]
        price = float(np.asarray(data.loc[t, "close"]).item())
        self._cache_price(ck, price)
        return price

    def _get_cached_price(self, key: Tuple[str, datetime]) -> Optional[float]:
        if key in self._price_cache:
            p, ct = self._price_cache[key]
            if abs((key[1] - ct).total_seconds()) <= 3600:
                return p
        return None

    def _cache_price(self, key: Tuple[str, datetime], price: float):
        self._price_cache[key] = (price, key[1])
        if len(self._price_cache) > 500:
            sorted_keys = sorted(self._price_cache.keys(), key=lambda x: x[1])
            for k in sorted_keys[: len(sorted_keys) // 4]:
                self._price_cache.pop(k, None)

    # ── OHLCV ──────────────────────────────────────────────────────────
    def get_spot_ohlcv(self, asset: str, interval: str, limit: int) -> pd.DataFrame:
        asset = self._normalize_asset(asset)
        if not self._validate_asset(asset):
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        if hasattr(self, "_backtest_interval"):
            interval = self._backtest_interval
        self._validate_timeframe(interval)
        self._validate_limit(limit)
        data = self._load_data(self._asset_to_spot_symbol(asset), interval)
        if data.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return data[data.index <= self.current_time].tail(limit)[["open", "high", "low", "close", "volume"]].copy()

    def get_futures_ohlcv(self, asset: str, interval: str, limit: int) -> pd.DataFrame:
        asset = self._normalize_asset(asset)
        if not self._validate_asset(asset):
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "fundingRate", "openInterest"])
        if hasattr(self, "_backtest_interval"):
            interval = self._backtest_interval
        self._validate_timeframe(interval)
        self._validate_limit(limit)
        data = self._load_data(self._asset_to_futures_symbol(asset), interval)
        if data.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "fundingRate", "openInterest"])
        cols = ["open", "high", "low", "close", "volume", "fundingRate", "openInterest"]
        present = [c for c in cols if c in data.columns]
        result = data[data.index <= self.current_time].tail(limit)[present].copy()
        for c in cols:
            if c not in result.columns:
                result[c] = np.nan
        return result[cols]

    # ── Funding & OI ───────────────────────────────────────────────────
    def get_funding_rate(self, asset: str = "BTC", timestamp: Optional[datetime] = None) -> float:
        if not self._validate_asset(asset):
            return 0.0
        symbol = self._asset_to_futures_symbol(asset)
        try:
            data = self._load_data(symbol, self._get_optimal_timeframe())
        except RuntimeError:
            return 0.0
        if data.empty or "fundingRate" not in data.columns:
            return 0.0
        ts = timestamp or self.current_time
        t = data.index.asof(ts)
        if pd.isna(t):
            return 0.0
        v = data.loc[t, "fundingRate"]
        return 0.0 if pd.isna(v) else float(np.asarray(v).item())

    def get_funding_rate_history(self, asset: str, limit: int) -> pd.DataFrame:
        if not self._validate_asset(asset):
            return pd.DataFrame(columns=["timestamp", "fundingRate"])
        self._validate_limit(limit)
        try:
            data = self._load_data(self._asset_to_futures_symbol(asset), self._get_optimal_timeframe())
        except RuntimeError:
            return pd.DataFrame(columns=["timestamp", "fundingRate"])
        if data.empty or "fundingRate" not in data.columns:
            return pd.DataFrame(columns=["timestamp", "fundingRate"])
        hist = data[data.index <= self.current_time].tail(limit)
        return pd.DataFrame({"timestamp": hist.index, "fundingRate": hist["fundingRate"]}).reset_index(drop=True).dropna()

    def get_open_interest(self, asset: str = "BTC", timestamp: Optional[datetime] = None) -> float:
        if not self._validate_asset(asset):
            return 0.0
        try:
            data = self._load_data(self._asset_to_futures_symbol(asset), self._get_optimal_timeframe())
        except RuntimeError:
            return 0.0
        if data.empty or "openInterest" not in data.columns:
            return 0.0
        ts = timestamp or self.current_time
        t = data.index.asof(ts)
        if pd.isna(t):
            return 0.0
        v = data.loc[t, "openInterest"]
        return 0.0 if pd.isna(v) else float(np.asarray(v).item())

    def get_open_interest_history(self, asset: str, limit: int) -> pd.DataFrame:
        if not self._validate_asset(asset):
            return pd.DataFrame(columns=["timestamp", "openInterest"])
        self._validate_limit(limit)
        try:
            data = self._load_data(self._asset_to_futures_symbol(asset), self._get_optimal_timeframe())
        except RuntimeError:
            return pd.DataFrame(columns=["timestamp", "openInterest"])
        if data.empty or "openInterest" not in data.columns:
            return pd.DataFrame(columns=["timestamp", "openInterest"])
        hist = data[data.index <= self.current_time].tail(limit)
        return pd.DataFrame({"timestamp": hist.index, "openInterest": hist["openInterest"]}).reset_index(drop=True).dropna()

    # ── Funding payment application ────────────────────────────────────
    def _get_next_funding_time(self, current_time: datetime, exchange: str) -> datetime:
        funding_hours = list(range(24)) if exchange == "hyperliquid" else [0, 8, 16]
        for h in funding_hours:
            if h > current_time.hour:
                return current_time.replace(hour=h, minute=0, second=0, microsecond=0)
        return current_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

    def apply_funding_payment(self, asset: str, position_size: float, leverage: int = 1):
        if position_size == 0:
            return
        fr = self.get_funding_rate(asset)
        mark = self.get_price(asset)
        if mark <= 0 or not np.isfinite(mark):
            return
        if fr is None or not np.isfinite(fr):
            fr = 0.0
        effective_fr = fr / 8.0 if self.exchange == "hyperliquid" else fr
        if not np.isfinite(effective_fr):
            effective_fr = 0.0
        payment = position_size * mark * effective_fr
        self.balances["USDC"] -= payment
        self.funding_payments.append({
            "time": int(self.current_time.timestamp()), "asset": asset,
            "position_size": position_size, "funding_rate": fr,
            "effective_funding_rate": effective_fr, "mark_price": mark,
            "funding_payment": payment, "exchange": self.exchange,
        })
        self.accumulated_funding_fee -= payment

    # ── Order management ───────────────────────────────────────────────
    def _generate_order_id(self) -> str:
        oid = str(self._next_order_id)
        self._next_order_id += 1
        return oid

    def _place_limit_order(self, asset: str, side: str, quantity: float, price: float, trading_type: str = "futures") -> Dict:
        oid = self._generate_order_id()
        leverage = 1
        if trading_type == "futures" and asset in self.futures_position_details:
            leverage = self.futures_position_details[asset].get("leverage", 1)
        order = {
            "order_id": oid, "asset": asset, "side": side, "order_type": "limit",
            "quantity": quantity, "price": price, "status": "pending",
            "filled_quantity": 0.0, "remaining_quantity": quantity,
            "created_time": self.current_time, "filled_time": None,
            "trading_type": trading_type, "leverage": leverage,
            "filled_price": None, "fees_paid": 0.0,
        }
        self.pending_orders[oid] = order
        return order

    def _should_execute_order(self, order: Dict, current_price: float, high: Optional[float] = None, low: Optional[float] = None) -> bool:
        if order["order_type"] == "market":
            return True
        lo = low if low is not None else current_price
        hi = high if high is not None else current_price
        if order["side"] in ("buy", "long"):
            return lo <= order["price"]
        if order["side"] in ("sell", "short"):
            return hi >= order["price"]
        return False

    def _process_pending_orders(self):
        if not self.pending_orders:
            return
        assets = {o["asset"] for o in self.pending_orders.values() if o["status"] == "pending"}
        for asset in assets:
            try:
                cp = self.get_price(asset)
                hi, lo = cp, cp
                try:
                    sym = self._asset_to_futures_symbol(asset)
                    ck = (sym, self._get_optimal_timeframe())
                    if ck in self.data_cache:
                        d = self.data_cache[ck]
                        if not d.empty:
                            t = d.index.asof(self.current_time)
                            if not pd.isna(t):
                                row = d.loc[t]
                                hi, lo, cp = float(row["high"]), float(row["low"]), float(row["close"])
                except Exception:
                    pass
                if math.isnan(cp) or cp <= 0:
                    continue
                for o in sorted(
                    [v for v in self.pending_orders.values() if v["asset"] == asset and v["status"] == "pending"],
                    key=lambda x: x["created_time"],
                ):
                    if self._should_execute_order(o, cp, hi, lo):
                        fp = o["price"] if o["order_type"] != "market" else cp
                        self._execute_order(o, fp)
            except Exception:
                pass

    def _execute_order(self, order: Dict, execution_price: float):
        try:
            oid = order["order_id"]
            asset, side, qty = order["asset"], order["side"], order["remaining_quantity"]
            tt, lev = order["trading_type"], order.get("leverage", 1)
            nv = execution_price * qty
            fee = nv * self.fee_rate
            rpnl = 0.0
            if tt == "futures":
                rpnl = self._futures_trade_execution(asset, qty, execution_price, "long" if side == "long" else "short", lev)
            else:
                if side == "buy":
                    self._spot_trade_execution(asset, qty, execution_price, "buy")
                else:
                    self._spot_trade_execution(asset, qty, execution_price, "sell")
            order.update({"status": "filled", "filled_quantity": qty, "remaining_quantity": 0.0,
                          "filled_time": int(self.current_time.timestamp()), "filled_price": execution_price, "fees_paid": fee})
            self.filled_orders.append(order.copy())
            del self.pending_orders[oid]
            if ENABLE_STRUCTURED_LOGGING:
                avg_cost = execution_price
                if asset in self.futures_position_details:
                    avg_cost = self.futures_position_details[asset].get("avg_entry_price", execution_price)
                log_trade_execution(action=f"{side}_order_executed", asset=asset, quantity=qty,
                                    price=execution_price, value=nv,
                                    timestamp=self.current_time.isoformat() if self.current_time else None,
                                    pnl=rpnl, position_avg_cost=avg_cost, fee=fee)
        except Exception as e:
            logger.error(f"Order execution failed: {e}")

    # ── Trade execution core ───────────────────────────────────────────
    def _futures_trade_execution(self, asset: str, quantity: float, price: float, action: str, leverage: int = 1) -> float:
        pm = self.perp_markets.get(asset)
        sd = pm["sz_decimals"] if pm else 4
        pd_ = pm["price_decimals"] if pm else 4
        quantity = float(f"{quantity:.{sd}f}")
        price = float(f"{price:.{pd_}f}")
        nv = price * quantity
        fee = nv * self.fee_rate

        if asset not in self.futures_position_details:
            self.futures_position_details[asset] = {"net_size": 0.0, "avg_entry_price": 0.0, "total_margin_used": 0.0, "trades": [], "leverage": leverage}
        det = self.futures_position_details[asset]
        cur = det["net_size"]
        is_long = action == "long"
        is_opp = (cur != 0) and ((is_long and cur < 0) or (not is_long and cur > 0))

        reduce_sz = float(f"{min(abs(cur), quantity):.{sd}f}") if is_opp else 0.0
        open_sz = max(0.0, quantity - reduce_sz)
        margin_req = (open_sz * price) / leverage if open_sz > 0 else 0.0

        if self.balances["USDC"] < margin_req + fee:
            print(f"Insufficient margin for {action} {quantity} {asset}")
            return 0.0

        if margin_req > 0:
            det["total_margin_used"] += margin_req
            self.locked_margin += margin_req
            self.balances["USDC"] -= margin_req

        released, rpnl = 0.0, 0.0
        if reduce_sz > 0:
            ep = det["avg_entry_price"]
            rpnl = ((price - ep) if cur > 0 else (ep - price)) * reduce_sz
            new_after = cur - reduce_sz if cur > 0 else cur + reduce_sz
            released = (reduce_sz * price) / max(1, det["leverage"])
            det["total_margin_used"] = max(0.0, det["total_margin_used"] - released)
            self.locked_margin = max(0.0, self.locked_margin - released)
            if abs(new_after) < 1e-12:
                det["avg_entry_price"] = 0.0
            cur = new_after

        ns = cur
        if open_sz > 0:
            if is_long:
                if ns > 0:
                    t_not = abs(ns) * det["avg_entry_price"] + open_sz * price
                    det["avg_entry_price"] = t_not / (abs(ns) + open_sz) if (abs(ns) + open_sz) > 0 else price
                else:
                    det["avg_entry_price"] = price
                ns += open_sz
            else:
                if ns < 0:
                    t_not = abs(ns) * det["avg_entry_price"] + open_sz * price
                    det["avg_entry_price"] = t_not / (abs(ns) + open_sz) if (abs(ns) + open_sz) > 0 else price
                else:
                    det["avg_entry_price"] = price
                ns -= open_sz

        det["net_size"] = float(f"{ns:.{sd}f}")
        det["leverage"] = leverage
        self.balances["USDC"] -= fee
        self.total_tx_fees += fee
        if rpnl != 0:
            self.balances["USDC"] += rpnl
        if released > 0:
            self.balances["USDC"] += released

        self.futures_positions[asset] = float(f"{ns:.{sd}f}")
        self.trades.append({
            "time": int(self.current_time.timestamp()), "action": action, "asset": asset,
            "quantity": quantity, "price": price, "leverage": leverage, "notional_value": nv,
            "required_margin": margin_req, "released_margin": released,
            "realized_pnl": rpnl, "fee": fee, "type": "futures",
            "avg_entry_price": det["avg_entry_price"], "net_position_size": ns,
            "reduce_size": reduce_sz, "open_size": open_sz,
        })
        return rpnl

    def _spot_trade_execution(self, asset: str, quantity: float, price: float, action: str):
        sm = self.spot_markets.get(asset)
        sd = sm["sz_decimals"] if sm else 4
        pd_ = sm["price_decimals"] if sm else 4
        quantity = float(f"{quantity:.{sd}f}")
        price = float(f"{price:.{pd_}f}")

        if action == "buy":
            cost = price * quantity
            if self.balances["USDC"] < cost:
                print(f"Insufficient USDC for buying {quantity} {asset}")
                return
            fee_asset = quantity * self.fee_rate
            received = quantity - fee_asset
            fee_usd = fee_asset * price
            self.balances["USDC"] -= cost
            self.balances[asset] = self.balances.get(asset, 0.0) + received
            self.total_tx_fees += fee_usd

            if asset not in self.position_tracking:
                self.position_tracking[asset] = {"quantity": 0.0, "avg_cost": 0.0, "total_cost": 0.0}
            pos = self.position_tracking[asset]
            new_qty = pos["quantity"] + received
            new_tc = pos["total_cost"] + cost
            pos.update({"quantity": new_qty, "total_cost": new_tc, "avg_cost": new_tc / new_qty if new_qty > 0 else 0.0})

            self.trades.append({
                "time": int(self.current_time.timestamp()), "action": "buy", "asset": asset,
                "quantity": quantity, "received_quantity": received, "price": price,
                "cost": cost, "fee": fee_usd, "pnl": 0.0, "type": "spot",
                "position_avg_cost": pos["avg_cost"], "position_quantity": new_qty,
            })
        elif action == "sell":
            avail = self.balances.get(asset, 0.0)
            if avail < quantity:
                print(f"Insufficient {asset} for selling {quantity}")
                return
            val = price * quantity
            fee = val * self.fee_rate
            net = val - fee
            pnl = 0.0
            avg_cost = 0.0
            if asset in self.position_tracking:
                pos = self.position_tracking[asset]
                avg_cost = pos["avg_cost"]
                if avg_cost > 0:
                    pnl = (price - avg_cost) * quantity - fee
                    old_q = pos["quantity"]
                    if old_q >= quantity:
                        rem = old_q - quantity
                        cost_sold = (quantity / old_q) * pos["total_cost"] if old_q > 0 else 0
                        pos.update({
                            "quantity": rem,
                            "total_cost": pos["total_cost"] - cost_sold,
                            "avg_cost": (pos["total_cost"] - cost_sold) / rem if rem > 0 else 0.0,
                        })
            self.balances[asset] -= quantity
            self.balances["USDC"] += net
            self.total_tx_fees += fee
            self.trades.append({
                "time": int(self.current_time.timestamp()), "action": "sell", "asset": asset,
                "quantity": quantity, "price": price, "value": val, "fee": fee,
                "net_value": net, "pnl": pnl, "type": "spot",
                "position_avg_cost": avg_cost,
            })

    # ── Public trading API ─────────────────────────────────────────────
    def buy(self, asset: str, quantity: float, price: float, order_type: str = "limit") -> Dict[str, Any]:
        if asset.upper() == "USDC":
            return SpotOrderResponse.Error("Cannot buy USDC").to_dict()
        if not self._validate_asset(asset):
            return SpotOrderResponse.Error(f"Asset {asset} not supported").to_dict()
        if quantity <= 0:
            return SpotOrderResponse.Error(f"Quantity must be positive").to_dict()
        if order_type == "market":
            cp = self.get_price(asset)
            if math.isnan(cp) or cp <= 0:
                return SpotOrderResponse.Error(f"Invalid price for {asset}").to_dict()
            self._spot_trade_execution(asset, quantity, cp, "buy")
            order = {"order_id": self._generate_order_id(), "asset": asset, "side": "buy",
                     "order_type": "market", "quantity": quantity, "price": cp,
                     "created_time": self.current_time}
            self.filled_orders.append(order)
        else:
            order = self._place_limit_order(asset, "buy", quantity, price, "spot")
        oi = SpotOrder(id=order["order_id"], asset=asset, side="buy", type=order["order_type"],
                       size=quantity, price=order["price"],
                       timestamp=int(self.current_time.timestamp())).to_dict()
        return SpotOrderResponse(status="success", order=oi).to_dict()

    def sell(self, asset: str, quantity: float, price: float, order_type: str = "limit") -> Dict[str, Any]:
        if asset.upper() == "USDC":
            return SpotOrderResponse.Error("Cannot sell USDC").to_dict()
        if not self._validate_asset(asset):
            return SpotOrderResponse.Error(f"Asset {asset} not supported").to_dict()
        if quantity <= 0:
            return SpotOrderResponse.Error(f"Quantity must be positive").to_dict()
        if order_type == "market":
            cp = self.get_price(asset)
            if math.isnan(cp) or cp <= 0:
                return SpotOrderResponse.Error(f"Invalid price for {asset}").to_dict()
            self._spot_trade_execution(asset, quantity, cp, "sell")
            order = {"order_id": self._generate_order_id(), "asset": asset, "side": "sell",
                     "order_type": "market", "quantity": quantity, "price": cp,
                     "created_time": self.current_time}
            self.filled_orders.append(order)
        else:
            order = self._place_limit_order(asset, "sell", quantity, price, "spot")
        oi = SpotOrder(id=order["order_id"], asset=asset, side="sell", type=order["order_type"],
                       size=quantity, price=order["price"],
                       timestamp=int(self.current_time.timestamp())).to_dict()
        return SpotOrderResponse(status="success", order=oi).to_dict()

    def long(self, asset: str, quantity: float, price: float, order_type: str = "limit") -> Dict[str, Any]:
        if not self._validate_asset(asset):
            return PerpOrderResponse.Error(f"Asset {asset} not supported").to_dict()
        if quantity <= 0:
            return PerpOrderResponse.Error("Quantity must be positive").to_dict()
        lev = self.futures_position_details.get(asset, {}).get("leverage", 1)
        if order_type == "market":
            cp = self.get_price(asset)
            if math.isnan(cp) or cp <= 0:
                return PerpOrderResponse.Error(f"Invalid price for {asset}").to_dict()
            self._futures_trade_execution(asset, quantity, cp, "long", lev)
            order = {"order_id": self._generate_order_id(), "asset": asset, "side": "long",
                     "order_type": "market", "quantity": quantity, "price": cp,
                     "created_time": self.current_time}
            self.filled_orders.append(order)
        else:
            order = self._place_limit_order(asset, "long", quantity, price, "futures")
        oi = PerpOrder(id=order["order_id"], asset=asset, side="long", type=order["order_type"],
                       size=quantity, price=order["price"],
                       timestamp=int(self.current_time.timestamp())).to_dict()
        return PerpOrderResponse(status="success", order=oi).to_dict()

    def short(self, asset: str, quantity: float, price: float, order_type: str = "limit") -> Dict[str, Any]:
        if not self._validate_asset(asset):
            return PerpOrderResponse.Error(f"Asset {asset} not supported").to_dict()
        if quantity <= 0:
            return PerpOrderResponse.Error("Quantity must be positive").to_dict()
        lev = self.futures_position_details.get(asset, {}).get("leverage", 1)
        if order_type == "market":
            cp = self.get_price(asset)
            if math.isnan(cp) or cp <= 0:
                return PerpOrderResponse.Error(f"Invalid price for {asset}").to_dict()
            self._futures_trade_execution(asset, quantity, cp, "short", lev)
            order = {"order_id": self._generate_order_id(), "asset": asset, "side": "short",
                     "order_type": "market", "quantity": quantity, "price": cp,
                     "created_time": self.current_time}
            self.filled_orders.append(order)
        else:
            order = self._place_limit_order(asset, "short", quantity, price, "futures")
        oi = PerpOrder(id=order["order_id"], asset=asset, side="short", type=order["order_type"],
                       size=quantity, price=order["price"],
                       timestamp=int(self.current_time.timestamp())).to_dict()
        return PerpOrderResponse(status="success", order=oi).to_dict()

    def reduce_position(self, asset: str, quantity: float) -> Dict[str, Any]:
        if not self._validate_asset(asset):
            return PerpOrderResponse.Error(f"Asset {asset} not supported").to_dict()
        if quantity <= 0:
            return PerpOrderResponse.Error("Quantity must be positive").to_dict()
        if asset not in self.futures_position_details:
            return PerpOrderResponse.Error(f"No position for {asset}").to_dict()
        det = self.futures_position_details[asset]
        ns = det["net_size"]
        if ns == 0:
            return PerpOrderResponse.Error(f"No position for {asset}").to_dict()
        close_qty = min(quantity, abs(ns))
        action = "short" if ns > 0 else "long"
        price = self.get_price(asset)
        ep = det["avg_entry_price"]
        rpnl = ((price - ep) if ns > 0 else (ep - price)) * close_qty
        nv = price * close_qty
        fee = nv * self.fee_rate
        proportion = close_qty / abs(ns)
        margin_freed = det["total_margin_used"] * proportion
        pm = self.perp_markets.get(asset)
        sd = pm["sz_decimals"] if pm else 4
        new_ns = ns - close_qty if ns > 0 else ns + close_qty
        det["net_size"] = float(f"{new_ns:.{sd}f}")
        det["total_margin_used"] -= margin_freed
        if abs(new_ns) < 1e-12:
            det["avg_entry_price"] = 0.0
            det["total_margin_used"] = 0.0
        self.balances["USDC"] += rpnl + margin_freed - fee
        self.locked_margin = max(0.0, self.locked_margin - margin_freed)
        self.total_tx_fees += fee
        self.futures_positions[asset] = float(f"{new_ns:.{sd}f}")
        self.trades.append({
            "time": int(self.current_time.timestamp()), "action": action, "asset": asset,
            "quantity": close_qty, "price": price, "notional_value": nv,
            "realized_pnl": rpnl, "fee": fee, "type": "futures",
            "avg_entry_price": ep, "net_position_size": new_ns,
            "reduce_size": close_qty, "open_size": 0.0,
        })
        if ENABLE_STRUCTURED_LOGGING:
            log_trade_execution(action="reduce_position", asset=asset, quantity=close_qty,
                                price=price, value=nv,
                                timestamp=self.current_time.isoformat() if self.current_time else None,
                                pnl=rpnl, position_avg_cost=ep, fee=fee)
        oi = PerpOrder(id=self._generate_order_id(), asset=asset, side=action, type="market",
                       size=close_qty, price=price, timestamp=int(self.current_time.timestamp())).to_dict()
        return PerpOrderResponse(status="success", order=oi).to_dict()

    def set_leverage(self, asset: str, leverage: int):
        self._validate_leverage(leverage)
        if asset not in self.futures_position_details:
            self.futures_position_details[asset] = {"net_size": 0.0, "avg_entry_price": 0.0, "total_margin_used": 0.0, "trades": [], "leverage": leverage}
        else:
            det = self.futures_position_details[asset]
            if abs(det["net_size"]) < 1e-8:
                det["leverage"] = leverage
            else:
                new_margin = abs(det["net_size"]) * det["avg_entry_price"] / leverage
                diff = new_margin - det["total_margin_used"]
                if diff > 0 and self.balances["USDC"] < diff:
                    raise RuntimeError(f"Insufficient margin to set leverage {leverage}x for {asset}")
                self.balances["USDC"] -= diff
                self.locked_margin += diff
                det["total_margin_used"] = new_margin
                det["leverage"] = leverage

    def get_futures_position(self, asset: str) -> float:
        return self.futures_positions.get(asset, 0.0)

    # ── Account summaries ──────────────────────────────────────────────
    def get_futures_unrealized_pnl(self, asset: Optional[str] = None) -> Union[float, Dict[str, float]]:
        if asset is not None:
            if asset not in self.futures_position_details:
                return 0.0
            det = self.futures_position_details[asset]
            ns = det["net_size"]
            if abs(ns) < 1e-8:
                return 0.0
            try:
                cp = self.get_price(asset)
                ep = det["avg_entry_price"]
                return (cp - ep) * ns if ep > 0 else 0.0
            except Exception:
                return 0.0
        return {a: self.get_futures_unrealized_pnl(a) for a in self.futures_position_details if self.get_futures_unrealized_pnl(a) != 0}

    def get_total_futures_unrealized_pnl(self) -> float:
        total = 0.0
        for a in self.futures_position_details:
            try:
                v = self.get_futures_unrealized_pnl(a)
                if isinstance(v, (int, float)) and np.isfinite(v):
                    total += v
            except Exception:
                pass
        return total

    def get_available_margin(self) -> float:
        return self.balances.get("USDC", 0.0)

    def get_total_margin_used(self) -> float:
        return self.locked_margin

    def get_perp_summary(self) -> Dict[str, Any]:
        upnl = self.get_total_futures_unrealized_pnl()
        av = self.balances.get("USDC", 0.0) + upnl
        summary = PerpAccountSummary(
            time=int(self.current_time.timestamp()) if self.current_time else None,
            account_value=av, available_margin=self.get_available_margin(),
            total_margin_used=self.locked_margin, total_unrealized_pnl=upnl,
        )
        positions = []
        for a, det in self.futures_position_details.items():
            ns = det.get("net_size", 0.0)
            if abs(ns) < 1e-8:
                continue
            cp = self.get_price(a)
            positions.append(PerpPositionSummary(
                asset=a, size=ns, entry_price=det.get("avg_entry_price", 0.0),
                unrealized_pnl=self.get_futures_unrealized_pnl(a),
                position_value=abs(ns) * cp if cp else 0.0,
                margin_used=det.get("total_margin_used", 0.0),
            ))
        summary.positions = positions
        return summary.to_dict()

    def get_perp_position(self, asset: str) -> Optional[Dict[str, Any]]:
        for p in self.get_perp_summary().get("positions", []):
            if p.get("asset") == asset:
                return p
        return None

    def get_spot_summary(self) -> Dict[str, Any]:
        bals = [SpotBalanceSummary(asset=a, total=b, free=b, locked=0) for a, b in self.balances.items()]
        return SpotAccountSummary(
            time=int(self.current_time.timestamp()) if self.current_time else None,
            balances=bals,
        ).to_dict()

    # ── Order queries ──────────────────────────────────────────────────
    def get_perp_open_orders(self, asset: Optional[str] = None) -> List[Dict[str, Any]]:
        orders = []
        for o in self.pending_orders.values():
            if o["status"] == "pending" and o["trading_type"] == "futures" and (asset is None or o["asset"] == asset):
                po = PerpOrder(id=o["order_id"], asset=o["asset"], side=o["side"],
                               type=o["order_type"], size=o["quantity"], price=o["price"],
                               timestamp=int(o["created_time"].timestamp())).to_dict()
                orders.append(po)
        return orders

    def get_spot_open_orders(self, asset: Optional[str] = None) -> List[Dict[str, Any]]:
        orders = []
        for o in self.pending_orders.values():
            if o["status"] == "pending" and o["trading_type"] == "spot" and (asset is None or o["asset"] == asset):
                so = SpotOrder(id=o["order_id"], asset=o["asset"], side=o["side"],
                               type=o["order_type"], size=o["quantity"], price=o["price"],
                               timestamp=int(o["created_time"].timestamp())).to_dict()
                orders.append(so)
        return orders

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.pending_orders:
            o = self.pending_orders[order_id]
            o["status"] = "cancelled"
            self.cancelled_orders.append(o.copy())
            del self.pending_orders[order_id]
            return True
        return False

    def _cancel_orders(self, asset: str, order_ids: List[str]) -> Dict[str, Any]:
        result = []
        for oid in order_ids:
            if self.cancel_order(oid):
                result.append({"status": "success", "id": oid})
            else:
                result.append({"status": "error", "id": oid, "error": f"Order {oid} not found"})
        return CancelOrdersResponse(status="success", orders=result).to_dict()

    def cancel_spot_orders(self, asset: str, order_ids: List[str]) -> Dict[str, Any]:
        return self._cancel_orders(asset, order_ids)

    def cancel_perp_orders(self, asset: str, order_ids: List[str]) -> Dict[str, Any]:
        return self._cancel_orders(asset, order_ids)

    # ── Time ───────────────────────────────────────────────────────────
    def get_current_time(self) -> datetime:
        return self.current_time

    def set_backtest_interval(self, interval: str):
        self._backtest_interval = interval

    def advance_time(self, interval: str) -> Optional[datetime]:
        if self.current_time is None:
            return None
        prev = self.current_time
        try:
            if interval.endswith("m"):
                new_t = self.current_time + timedelta(minutes=int(interval[:-1]))
            elif interval.endswith("h"):
                new_t = self.current_time + timedelta(hours=int(interval[:-1]))
            elif interval.endswith("d"):
                new_t = self.current_time + timedelta(days=int(interval[:-1]))
            else:
                return None
        except ValueError:
            return None

        end_dt = datetime.strptime(self.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if new_t.date() > end_dt.date():
            return None

        # Apply funding payments
        nft = self._get_next_funding_time(prev, self.exchange)
        while nft <= new_t:
            orig = self.current_time
            self.current_time = nft
            for a, det in self.futures_position_details.items():
                ps = det.get("net_size", 0)
                if ps != 0:
                    self.apply_funding_payment(a, ps, det.get("leverage", 1))
            self.current_time = orig
            nft = self._get_next_funding_time(nft, self.exchange)

        # Manage price cache
        stale = [k for k in self._price_cache if k[1] < new_t - timedelta(hours=1)]
        for k in stale:
            self._price_cache.pop(k, None)
        if len(self._price_cache) > 1000:
            for k in sorted(self._price_cache, key=lambda x: x[1])[: len(self._price_cache) // 2]:
                self._price_cache.pop(k, None)

        self.current_time = new_t
        self._process_pending_orders()
        return new_t
