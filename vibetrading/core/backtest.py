"""
BacktestEngine - Main orchestrator for the trading strategy backtesting system.

Coordinates the backtesting process: sandbox creation, strategy code execution,
time stepping, metrics collection, and result aggregation.
"""

import importlib
import importlib.util
import sys
import traceback
import re
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .sandbox_base import SUPPORTED_INTERVALS
from .static_sandbox import StaticSandbox
from .error_handler import StrategyErrorHandler
from ..metrics.calculator import MetricsCalculator
from ..utils.logging import (
    log_portfolio_update,
    log_pnl_update,
    log_metrics_update,
    log_portfolio_composition,
    log_runtime_error,
)

logger = logging.getLogger(__name__)


def strip_ansi_colors(text: str) -> str:
    """Remove ANSI color codes from text."""
    return re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])').sub('', text)


class _SelectiveOutput:
    """Custom output stream that only allows structured logging prefixes to pass."""
    PREFIXES = [
        'TRADE_EXECUTED:', 'PORTFOLIO_UPDATE:', 'PNL_UPDATE:',
        'METRICS_UPDATE:', 'PORTFOLIO_COMPOSITION:', 'STRATEGY_EVENT:',
        'RUNTIME_ERROR:', 'DOWNLOAD_PROGRESS:', 'DOWNLOAD_SUCCESS:',
        'DOWNLOAD_ERROR:', 'DOWNLOAD_WARNING:',
    ]

    def __init__(self, original):
        self.original = original

    def write(self, data):
        if any(p in data for p in self.PREFIXES):
            self.original.write(data)

    def flush(self):
        self.original.flush()


class _SuppressOutput:
    """Context manager to suppress stdout/stderr except structured logging."""
    def __init__(self, mute: bool = True):
        self.mute = mute
        self._orig_out = None
        self._orig_err = None

    def __enter__(self):
        if self.mute:
            self._orig_out = sys.stdout
            self._orig_err = sys.stderr
            sys.stdout = _SelectiveOutput(self._orig_out)
            sys.stderr = _SelectiveOutput(self._orig_err)
        return self

    def __exit__(self, *args):
        if self.mute and self._orig_out:
            sys.stdout = self._orig_out
            sys.stderr = self._orig_err


class BacktestEngine:
    """Main orchestrator class for the backtesting system."""

    def __init__(
        self,
        initial_balances: Optional[Dict[str, float]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        interval: str = "1h",
        exchange: str = "binance",
        mute_strategy_prints: bool = False,
        data: Optional[Dict] = None,
    ):
        """
        Initialize the backtesting environment.

        Args:
            initial_balances: Initial balances for different assets.
            start_time: Start time for the backtest.
            end_time: End time for the backtest.
            interval: Time interval (e.g., "5m", "1h", "1d").
            exchange: Exchange name (e.g., 'binance', 'hyperliquid').
            mute_strategy_prints: Suppress print statements from strategy code.
            data: Pre-loaded data dict mapping
                  ``"ASSET/interval"`` keys to DataFrames. When provided,
                  the sandbox skips CSV lookup and uses this data directly.
        """
        if interval not in SUPPORTED_INTERVALS:
            raise ValueError(f"Unsupported interval: {interval}. Supported: {list(SUPPORTED_INTERVALS.keys())}")

        self.interval = interval
        self._initial_balances_config = initial_balances
        self.exchange = exchange.lower()
        self.mute_strategy_prints = mute_strategy_prints

        self.start_time = start_time or datetime(2025, 1, 1, tzinfo=timezone.utc)
        self.end_time = end_time or (self.start_time + timedelta(days=180))

        start_str = self.start_time.strftime('%Y-%m-%d')
        end_str = self.end_time.strftime('%Y-%m-%d')

        # Convert "ASSET/interval" keyed data to (symbol, interval) keyed data
        sandbox_data = None
        if data is not None:
            sandbox_data = self._convert_data_keys(data, interval)

        self.sandbox = StaticSandbox(
            exchange=self.exchange, start_date=start_str, end_date=end_str,
            initial_balances=self._initial_balances_config,
            mute_strategy_prints=self.mute_strategy_prints,
            data=sandbox_data,
        )
        self.sandbox.set_backtest_interval(self.interval)

        self.results: List[Dict[str, Any]] = []
        self.registered_strategy_callbacks: List[Callable] = []

        self._initial_total_value: Optional[float] = None
        self._all_time_peak: Optional[float] = None
        self._max_drawdown_observed: Optional[float] = None
        self._last_recorded_trade_count = 0
        self._liquidated = False
        self._liquidation_time = None
        self._liquidation_step = None

        self._win_rate_tracker = {
            'winning_trades': 0, 'losing_trades': 0,
            'last_processed_trade_index': 0, 'total_closed_trades': 0,
        }
        self._sharpe_tracker = {
            'return_count': 0, 'mean': 0.0, 'm2': 0.0, 'last_total_value': None,
        }

        self.metrics_calculator = MetricsCalculator(self)

    @staticmethod
    def _convert_data_keys(data: Dict, interval: str) -> Dict:
        """Convert 'ASSET/interval' keyed data to '(symbol, interval)' for sandbox."""
        from ..tools.data_loader import DEFAULT_PERP_SYMBOLS, DEFAULT_SPOT_SYMBOLS
        converted = {}
        for key, df in data.items():
            if isinstance(key, tuple):
                converted[key] = df
                continue
            # Parse "BTC/1h" -> asset="BTC", tf="1h"
            parts = key.split("/")
            asset = parts[0].upper()
            tf = parts[1] if len(parts) > 1 else interval
            # Map to exchange symbols
            if asset in DEFAULT_PERP_SYMBOLS:
                converted[(DEFAULT_PERP_SYMBOLS[asset], tf)] = df
            if asset in DEFAULT_SPOT_SYMBOLS:
                converted[(DEFAULT_SPOT_SYMBOLS[asset], tf)] = df
        return converted

    # ── Interval helpers ───────────────────────────────────────────────
    def get_interval_minutes(self, interval: str) -> float:
        mapping = {"1s": 1/60, "1m": 1, "5m": 5, "15m": 15, "30m": 30,
                   "1h": 60, "4h": 240, "1d": 1440, "1w": 10080, "1M": 43200}
        if interval not in mapping:
            raise ValueError(f"Unsupported interval: {interval}")
        return mapping[interval]

    # ── Streaming control ──────────────────────────────────────────────
    def _should_stream_12hourly(self) -> bool:
        if not hasattr(self, '_last_stream_12hour'):
            if self.sandbox.current_time:
                h = self.sandbox.current_time.hour
                base_h = 0 if h < 12 else 12
                self._last_stream_12hour = self.sandbox.current_time.replace(hour=base_h, minute=0, second=0, microsecond=0)
                return True
            return False
        if self.sandbox.current_time:
            h = self.sandbox.current_time.hour
            cur = self.sandbox.current_time.replace(hour=(0 if h < 12 else 12), minute=0, second=0, microsecond=0)
            if cur != self._last_stream_12hour:
                self._last_stream_12hour = cur
                return True
        return False

    def _should_stream_metrics(self) -> bool:
        if not hasattr(self, '_last_metrics_update'):
            if self.sandbox.current_time:
                self._last_metrics_update = self.sandbox.current_time
                return True
            return False
        if not self.sandbox.current_time:
            return False
        days = (self.end_time - self.start_time).days
        if days < 30:
            self._last_metrics_update = self.sandbox.current_time
            return True
        elif days <= 90:
            if self.sandbox.current_time.date() != self._last_metrics_update.date():
                self._last_metrics_update = self.sandbox.current_time
                return True
            return False
        else:
            return self._should_stream_12hourly()

    # ── Time initialization ────────────────────────────────────────────
    def _initialize_sandbox_time(self):
        if self.sandbox.current_time < self.start_time:
            self.sandbox.current_time = self.start_time
        try:
            self._initial_total_value = self._calculate_total_value_safe()
        except Exception:
            self._initial_total_value = self.sandbox.balances.get("USDC", 0.0)
        self._all_time_peak = self._initial_total_value
        self._max_drawdown_observed = 0.0
        self._sharpe_tracker['last_total_value'] = self._initial_total_value

    def _calculate_total_value_safe(self) -> float:
        total = self.sandbox.balances.get("USDC", 0.0)
        for asset, amount in self.sandbox.balances.items():
            if asset != "USDC" and amount > 0:
                try:
                    total += amount * self.sandbox.get_price(asset)
                except Exception:
                    pass
        return total

    def _calculate_total_value(self) -> float:
        total = self.sandbox.balances.get("USDC", 0.0)
        for asset, amount in self.sandbox.balances.items():
            if asset != "USDC" and amount > 0:
                try:
                    total += amount * self.sandbox.get_price(asset)
                except Exception:
                    pass
        try:
            if hasattr(self.sandbox, 'locked_margin'):
                total += self.sandbox.locked_margin
        except Exception:
            pass
        try:
            if hasattr(self.sandbox, 'get_total_futures_unrealized_pnl'):
                total += self.sandbox.get_total_futures_unrealized_pnl()
        except Exception:
            pass
        return total

    # ── Incremental trackers ───────────────────────────────────────────
    def _update_win_rate_tracker_incremental(self):
        if not hasattr(self.sandbox, 'trades'):
            return
        cur = len(self.sandbox.trades)
        last = self._win_rate_tracker['last_processed_trade_index']
        for t in self.sandbox.trades[last:cur]:
            pnl = None
            if t.get('action') in ('sell', 'close_position') and 'pnl' in t:
                pnl = t.get('pnl', 0.0)
            elif 'realized_pnl' in t:
                pnl = t.get('realized_pnl', 0.0)
            if pnl is not None:
                self._win_rate_tracker['total_closed_trades'] += 1
                if pnl > 0:
                    self._win_rate_tracker['winning_trades'] += 1
                elif pnl < 0:
                    self._win_rate_tracker['losing_trades'] += 1
        self._win_rate_tracker['last_processed_trade_index'] = cur

    def _update_sharpe_tracker_incremental(self, value: float):
        t = self._sharpe_tracker
        if t['last_total_value'] is None:
            t['last_total_value'] = value
            return
        prev = t['last_total_value']
        if prev > 0:
            ret = (value - prev) / prev
            t['return_count'] += 1
            n = t['return_count']
            d1 = ret - t['mean']
            t['mean'] += d1 / n
            d2 = ret - t['mean']
            t['m2'] += d1 * d2
        t['last_total_value'] = value

    # ── Result recording ───────────────────────────────────────────────
    def _record_results(self, timestamp: datetime):
        tv, comp = self._calculate_portfolio_combined()

        if self._all_time_peak is None:
            self._all_time_peak = tv
            self._max_drawdown_observed = 0.0
        elif tv > self._all_time_peak:
            self._all_time_peak = tv
        else:
            if self._all_time_peak > 0:
                dd = (tv - self._all_time_peak) / self._all_time_peak
                if self._max_drawdown_observed is None or dd < self._max_drawdown_observed:
                    self._max_drawdown_observed = dd

        dpnl = tv - self.results[-1]["total_value"] if self.results else 0.0
        self.results.append({
            "timestamp": timestamp, "balances": self.sandbox.balances.copy(),
            "equity": sum(self.sandbox.balances.values()),
            "total_value": tv, "daily_pnl": dpnl, "composition": comp,
        })
        self._last_recorded_trade_count = len(self.sandbox.trades)

        # Structured logging
        from .static_sandbox import ENABLE_STRUCTURED_LOGGING
        if ENABLE_STRUCTURED_LOGGING:
            if self._should_stream_metrics():
                m = self.metrics_calculator.calculate(use_all_data=True)
                ib = self._initial_total_value if self._initial_total_value else self.sandbox.INITIAL_USDC_BALANCE
                m['initial_balance'] = ib
                log_metrics_update(
                    win_rate=m.get('win_rate', 0.0), max_drawdown=m.get('max_drawdown', 0.0),
                    total_trades=m.get('total_trades', 0), sharpe_ratio=m.get('sharpe_ratio', 0.0),
                    total_return=m.get('total_return', 0.0), funding_revenue=m.get('funding_revenue', 0.0),
                    total_tx_fees=m.get('total_tx_fees', 0.0), initial_balance=ib,
                    timestamp=timestamp.isoformat(),
                )

    def _calculate_portfolio_combined(self) -> Tuple[float, Dict[str, float]]:
        tv = self.sandbox.balances.get("USDC", 0.0)
        comp: Dict[str, float] = {"USDC": tv}
        for a, b in self.sandbox.balances.items():
            if a != "USDC" and b > 0:
                try:
                    p = self.sandbox.get_price(a)
                    v = b * p
                    tv += v
                    comp[a] = v
                except Exception:
                    comp[a] = 0.0
        try:
            lm = getattr(self.sandbox, 'locked_margin', 0.0)
            tv += lm
            if lm > 0:
                comp["locked_margin"] = lm
        except Exception:
            pass
        try:
            fpnl = self.sandbox.get_total_futures_unrealized_pnl()
            tv += fpnl
            if fpnl != 0:
                comp["futures_unrealized_pnl"] = fpnl
        except Exception:
            pass
        return tv, comp

    # ── Main run ───────────────────────────────────────────────────────
    def run(self, strategy_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Run the backtest with the provided strategy code.

        Args:
            strategy_code: Python code containing the trading strategy with @vibe decorator

        Returns:
            Dictionary containing backtest results, metrics, and trade history
        """
        _orig_in_sys = 'vibetrading' in sys.modules
        _orig_mod = sys.modules.get('vibetrading')
        mock_mod = None

        try:
            if strategy_code:
                spec = importlib.util.spec_from_loader('vibetrading', loader=None)
                if spec is None:
                    raise ImportError("Cannot create vibetrading module spec")
                mock_mod = importlib.util.module_from_spec(spec)

                error_handler = StrategyErrorHandler(self.sandbox, strategy_code)

                def mock_vibe(interval_or_func=None, **kwargs):
                    if callable(interval_or_func):
                        func = interval_or_func
                        self.registered_strategy_callbacks.append(error_handler.wrap_strategy(func))
                        return func
                    else:
                        def dec(func):
                            self.registered_strategy_callbacks.append(error_handler.wrap_strategy(func))
                            return func
                        return dec

                exec_globals = {'pd': pd, 'np': np, 'ta': None}
                try:
                    import ta
                    exec_globals['ta'] = ta
                    sys.modules['ta'] = ta
                except ImportError:
                    pass

                funcs = {
                    'my_spot_balance': self.sandbox.my_spot_balance,
                    'my_futures_balance': self.sandbox.my_futures_balance,
                    'buy': self.sandbox.buy, 'sell': self.sandbox.sell,
                    'long': self.sandbox.long, 'short': self.sandbox.short,
                    'reduce_position': self.sandbox.reduce_position,
                    'set_leverage': self.sandbox.set_leverage,
                    'get_futures_position': self.sandbox.get_futures_position,
                    'get_price': self.sandbox.get_price,
                    'get_spot_ohlcv': self.sandbox.get_spot_ohlcv,
                    'get_futures_ohlcv': self.sandbox.get_futures_ohlcv,
                    'get_funding_rate': self.sandbox.get_funding_rate,
                    'get_funding_rate_history': self.sandbox.get_funding_rate_history,
                    'get_open_interest': self.sandbox.get_open_interest,
                    'get_open_interest_history': self.sandbox.get_open_interest_history,
                    'get_spot_price': self.sandbox.get_spot_price,
                    'get_perp_price': self.sandbox.get_perp_price,
                    'get_spot_summary': self.sandbox.get_spot_summary,
                    'get_perp_summary': self.sandbox.get_perp_summary,
                    'cancel_spot_orders': self.sandbox.cancel_spot_orders,
                    'cancel_perp_orders': self.sandbox.cancel_perp_orders,
                    'get_perp_position': self.sandbox.get_perp_position,
                    'get_current_time': self.sandbox.get_current_time,
                    'get_supported_assets': self.sandbox.get_supported_assets,
                    'get_spot_open_orders': self.sandbox.get_spot_open_orders,
                    'get_perp_open_orders': self.sandbox.get_perp_open_orders,
                    'cancel_order': self.sandbox.cancel_order,
                    'vibe': mock_vibe,
                }
                for name, obj in funcs.items():
                    setattr(mock_mod, name, obj)
                sys.modules['vibetrading'] = mock_mod
                exec_globals.update(funcs)

                exec(strategy_code, exec_globals)
                if not self.registered_strategy_callbacks:
                    raise ValueError("No strategy functions registered. Use @vibe decorator.")

            self._initialize_sandbox_time()

            current_time = self.sandbox.current_time
            step = 0
            max_steps = 30000

            while current_time is not None and current_time <= self.end_time and step < max_steps:
                step += 1

                if hasattr(self.sandbox, '_process_pending_orders'):
                    try:
                        self.sandbox._process_pending_orders()
                    except Exception:
                        pass

                if strategy_code and self.registered_strategy_callbacks:
                    for cb in self.registered_strategy_callbacks:
                        try:
                            with _SuppressOutput(mute=self.mute_strategy_prints):
                                cb()
                        except Exception as e:
                            tb = traceback.format_exc()
                            log_runtime_error(type(e).__name__, str(e), tb,
                                              getattr(cb, '__name__', 'callback'))
                            raise

                pv = self._calculate_total_value()
                self._update_sharpe_tracker_incremental(pv)

                if pv <= 0:
                    self._liquidated = True
                    self._liquidation_time = current_time
                    self._liquidation_step = step
                    self._record_results(current_time)
                    break

                self._record_results(current_time)
                prev = current_time
                current_time = self.sandbox.advance_time(self.interval)
                if current_time == prev:
                    break

            if strategy_code:
                final = self.get_metrics()
                tr = f"{self.start_time.strftime('%Y-%m-%d')} to {self.end_time.strftime('%Y-%m-%d')}"
                return {
                    'trades': self.sandbox.trades,
                    'final_balances': self.sandbox.balances.copy(),
                    'total_trades': len(self.sandbox.trades),
                    'results': self.get_results(),
                    'metrics': final,
                    'simulation_info': {
                        'steps': step,
                        'start_time': self.start_time.isoformat(),
                        'end_time': self.end_time.isoformat(),
                        'final_time': current_time.isoformat() if current_time else None,
                        'interval': self.interval,
                        'time_range': tr,
                        'liquidated': self._liquidated,
                        'liquidation_time': self._liquidation_time.isoformat() if self._liquidation_time else None,
                        'liquidation_step': self._liquidation_step,
                    },
                }
        except Exception as e:
            log_runtime_error(type(e).__name__, str(e), traceback.format_exc(), "backtest_run")
            raise
        finally:
            if mock_mod and 'vibetrading' in sys.modules and sys.modules['vibetrading'] is mock_mod:
                if _orig_in_sys and _orig_mod is not None:
                    sys.modules['vibetrading'] = _orig_mod
                elif not _orig_in_sys:
                    del sys.modules['vibetrading']

    def get_results(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    def get_metrics(self) -> Dict[str, Any]:
        if not self.results:
            return {}
        metrics = self.metrics_calculator.calculate(use_all_data=True)
        ib = self._initial_total_value if self._initial_total_value is not None else self.sandbox.INITIAL_USDC_BALANCE
        tv = self._calculate_total_value()
        df = pd.DataFrame(self.results)
        df["returns"] = df["total_value"].pct_change()
        df["cummax"] = df["total_value"].cummax()
        df["drawdown"] = (df["total_value"] - df["cummax"]) / df["cummax"]
        return {
            "total_return": metrics["total_return"],
            "number_of_trades": metrics["total_trades"],
            "max_drawdown": metrics["max_drawdown"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "win_rate": metrics["win_rate"],
            "funding_revenue": metrics["funding_revenue"],
            "total_tx_fees": metrics["total_tx_fees"],
            "average_trade_duration_hours": metrics["average_trade_duration_hours"],
            "total_value": tv,
            "initial_balance": ib,
            "pandas_max_drawdown": df["drawdown"].min(),
            "returns_std": df["returns"].std() if len(df["returns"]) > 1 else 0.0,
            "returns_mean": df["returns"].mean() if len(df["returns"]) > 1 else 0.0,
        }
