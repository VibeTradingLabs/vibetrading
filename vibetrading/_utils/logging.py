"""
Structured Logging for VibeTrading Backtests

Provides helper functions for backtest strategies to emit structured data
that can be parsed by backends and streamed live via WebSocket.

Includes a FIFO-based rate limiter to prevent frontend overload while
maintaining real-time streaming capabilities.
"""

import json
import time
from collections import deque
from datetime import datetime
from typing import Dict, Any, Optional


ENABLE_HUMAN_READABLE_LOGS = False


class FIFORateLimiter:
    """
    FIFO-based rate limiter using a sliding window approach.
    """

    def __init__(self, max_qps: float = 5.0, window_seconds: float = 1.0):
        self.max_qps = max_qps
        self.window_seconds = window_seconds
        self.max_events = int(max_qps * window_seconds)
        self.events = deque()
        self._stats = {
            'total_events': 0,
            'rate_limited_events': 0,
            'last_reset': time.time()
        }

    def should_allow(self, event_type: Optional[str] = None) -> bool:
        """Check if a new event should be allowed based on current rate."""
        current_time = time.time()
        self._stats['total_events'] += 1

        cutoff_time = current_time - self.window_seconds
        while self.events and self.events[0] <= cutoff_time:
            self.events.popleft()

        if len(self.events) < self.max_events:
            self.events.append(current_time)
            return True
        else:
            self._stats['rate_limited_events'] += 1
            return False

    def get_current_qps(self) -> float:
        """Get current QPS based on events in the window."""
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds
        while self.events and self.events[0] <= cutoff_time:
            self.events.popleft()
        return len(self.events) / self.window_seconds

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        current_time = time.time()
        uptime = current_time - self._stats['last_reset']
        return {
            'current_qps': self.get_current_qps(),
            'max_qps': self.max_qps,
            'events_in_window': len(self.events),
            'total_events': self._stats['total_events'],
            'rate_limited_events': self._stats['rate_limited_events'],
            'uptime_seconds': uptime
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self._stats = {
            'total_events': 0,
            'rate_limited_events': 0,
            'last_reset': time.time()
        }


_global_rate_limiter = FIFORateLimiter(max_qps=2.0, window_seconds=1.0)


def configure_rate_limiting(max_qps: float = 2.0, window_seconds: float = 1.0):
    """Configure global rate limiting parameters."""
    global _global_rate_limiter
    _global_rate_limiter = FIFORateLimiter(max_qps=max_qps, window_seconds=window_seconds)


def is_rate_limited(event_type: Optional[str] = None) -> bool:
    """Check if logging should be rate limited."""
    return not _global_rate_limiter.should_allow(event_type)


def get_rate_limiter_stats() -> Dict[str, Any]:
    """Get current rate limiter statistics."""
    return _global_rate_limiter.get_stats()


def log_portfolio_update(total_value, balances, futures_positions=None, timestamp=None,
                         balances_usd=None, futures_positions_usd=None):
    """Log a portfolio update in structured format."""
    if timestamp is None:
        timestamp = datetime.now().isoformat()

    data = {
        "timestamp": timestamp,
        "total_value": total_value,
        "balances": balances
    }

    if balances_usd:
        data["balances_usd"] = balances_usd

    if futures_positions:
        long_positions = {asset: size for asset, size in futures_positions.items() if size > 0}
        short_positions = {asset: abs(size) for asset, size in futures_positions.items() if size < 0}
        data["futures_positions"] = {"long": long_positions, "short": short_positions, "raw": futures_positions}
    else:
        data["futures_positions"] = {"long": {}, "short": {}, "raw": {}}

    if futures_positions_usd:
        long_usd = {a: v for a, v in futures_positions_usd.items() if v > 0}
        short_usd = {a: abs(v) for a, v in futures_positions_usd.items() if v < 0}
        data["futures_positions_usd"] = {"long": long_usd, "short": short_usd, "raw": futures_positions_usd}
    else:
        data["futures_positions_usd"] = {"long": {}, "short": {}, "raw": {}}

    print(f"PORTFOLIO_UPDATE:{json.dumps(data)}")


def log_trade_execution(action: str, asset: str, quantity: float, price: float,
                        value: Optional[float] = None, timestamp: Optional[str] = None,
                        pnl: Optional[float] = None, position_avg_cost: Optional[float] = None,
                        fee: Optional[float] = None):
    """Log a trade execution in structured format."""
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    if value is None or value == 0:
        value = quantity * price

    data = {
        "timestamp": timestamp,
        "action": action,
        "asset": asset,
        "quantity": quantity,
        "price": price,
        "value": value
    }
    if pnl is not None:
        data["pnl"] = pnl
    if position_avg_cost is not None:
        data["position_avg_cost"] = position_avg_cost
    if fee is not None:
        data["fee"] = fee

    print(f"TRADE_EXECUTED:{json.dumps(data)}")


def log_pnl_update(realized_pnl: float, unrealized_pnl: float, total_pnl: float,
                   funding_revenue_period: Optional[float] = None,
                   total_funding_revenue: Optional[float] = None,
                   timestamp: Optional[str] = None):
    """Log P&L update in structured format."""
    if timestamp is None:
        timestamp = datetime.now().isoformat()

    data = {
        "timestamp": timestamp,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "total_pnl": total_pnl
    }
    if funding_revenue_period is not None:
        data["funding_revenue_period"] = funding_revenue_period
    if total_funding_revenue is not None:
        data["total_funding_revenue"] = total_funding_revenue

    print(f"PNL_UPDATE:{json.dumps(data)}")

    if ENABLE_HUMAN_READABLE_LOGS:
        print(f"P&L Update: Realized ${realized_pnl:.2f}, Unrealized ${unrealized_pnl:.2f}, Total ${total_pnl:.2f}")


def log_strategy_event(event_type: str, message: str, data: Optional[Dict[str, Any]] = None,
                       timestamp: Optional[str] = None):
    """Log a custom strategy event in structured format."""
    if timestamp is None:
        timestamp = datetime.now().isoformat()

    event_data = {
        "timestamp": timestamp,
        "event_type": event_type,
        "message": message,
        "data": data or {}
    }
    print(f"STRATEGY_EVENT:{json.dumps(event_data)}")


def log_metrics_update(win_rate: float, max_drawdown: float, total_trades: int,
                       sharpe_ratio: float, total_return: float,
                       funding_revenue: float = 0.0,
                       total_tx_fees: float = 0.0,
                       initial_balance: float = 10000,
                       timestamp: Optional[str] = None):
    """Log metrics update in structured format."""
    if timestamp is None:
        timestamp = datetime.now().isoformat()

    data = {
        "timestamp": timestamp,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "total_trades": total_trades,
        "sharpe_ratio": sharpe_ratio,
        "total_return": total_return,
        "funding_revenue": funding_revenue,
        "total_tx_fees": total_tx_fees,
        "initial_balance": initial_balance
    }
    print(f"METRICS_UPDATE:{json.dumps(data)}")

    if ENABLE_HUMAN_READABLE_LOGS:
        print(
            f"Live Metrics: Return {total_return*100:.1f}%, Win Rate {win_rate*100:.1f}%, "
            f"Drawdown {max_drawdown*100:.1f}%, Trades {total_trades}, Sharpe {sharpe_ratio:.2f}"
        )


def log_portfolio_composition(composition: Dict[str, float], timestamp: Optional[str] = None):
    """Log portfolio composition in structured format."""
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    data = {"timestamp": timestamp, "composition": composition}
    print(f"PORTFOLIO_COMPOSITION:{json.dumps(data)}")


def log_runtime_error(error_type: str, error_message: str, error_traceback: Optional[str] = None,
                      strategy_function: Optional[str] = None, timestamp: Optional[str] = None,
                      strategy_code_context: Optional[Dict[str, Any]] = None):
    """Log a runtime error in structured format (not rate limited)."""
    if timestamp is None:
        timestamp = datetime.now().isoformat()

    data = {
        "timestamp": timestamp,
        "error_type": error_type,
        "error_message": error_message,
        "error_traceback": error_traceback,
        "strategy_function": strategy_function,
        "regeneration_needed": True
    }
    if strategy_code_context:
        data["strategy_code_context"] = strategy_code_context

    print(f"RUNTIME_ERROR:{json.dumps(data)}")


def log_download_progress(message: str, data_type: str = "general",
                         progress_percent: Optional[float] = None,
                         items_processed: Optional[int] = None,
                         total_items: Optional[int] = None,
                         timestamp: Optional[str] = None):
    """Log download progress in structured format."""
    if timestamp is None:
        timestamp = datetime.now().isoformat()

    data: Dict[str, Any] = {"timestamp": timestamp, "message": message, "data_type": data_type}
    if progress_percent is not None:
        data["progress_percent"] = progress_percent
    if items_processed is not None:
        data["items_processed"] = items_processed
    if total_items is not None:
        data["total_items"] = total_items

    print(f"DOWNLOAD_PROGRESS:{json.dumps(data)}", flush=True)


def log_download_success(message: str, items_count: Optional[int] = None,
                        data_type: str = "general", timestamp: Optional[str] = None):
    """Log download success in structured format."""
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    data: Dict[str, Any] = {"timestamp": timestamp, "message": message, "data_type": data_type}
    if items_count is not None:
        data["items_count"] = items_count
    print(f"DOWNLOAD_SUCCESS:{json.dumps(data)}", flush=True)


def log_download_error(message: str, error_type: str = "general", timestamp: Optional[str] = None):
    """Log download error in structured format."""
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    data = {"timestamp": timestamp, "message": message, "error_type": error_type}
    print(f"DOWNLOAD_ERROR:{json.dumps(data)}", flush=True)


def log_download_warning(message: str, warning_type: str = "general", timestamp: Optional[str] = None):
    """Log download warning in structured format."""
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    data = {"timestamp": timestamp, "message": message, "warning_type": warning_type}
    print(f"DOWNLOAD_WARNING:{json.dumps(data)}", flush=True)
