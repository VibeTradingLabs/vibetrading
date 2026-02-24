"""
Backtesting engine for trading strategies.

Usage::

    import vibetrading.backtest

    # Full control via BacktestEngine
    engine = vibetrading.backtest.BacktestEngine(
        interval="1h",
        initial_balances={"USDC": 10000},
    )
    results = engine.run(strategy_code)

    # Or use the convenience function
    results = vibetrading.backtest.run(
        strategy_code,
        interval="1h",
        initial_balances={"USDC": 10000},
    )
"""

from datetime import datetime
from typing import Any, Dict, Optional

from ._core.backtest import BacktestEngine
from ._core.static_sandbox import StaticSandbox
from ._metrics.calculator import MetricsCalculator


def run(
    strategy_code: str,
    *,
    interval: str = "1h",
    initial_balances: Optional[Dict[str, float]] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    exchange: str = "binance",
    data: Optional[Dict] = None,
    mute_strategy_prints: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Run a backtest in one call.

    Args:
        strategy_code: Python code with a @vibe-decorated strategy function.
        interval: Candle interval (e.g. "5m", "1h", "1d").
        initial_balances: Starting balances (default: {"USDC": 10000}).
        start_time: Backtest start (default: 2025-01-01 UTC).
        end_time: Backtest end (default: 180 days after start).
        exchange: Exchange name for data lookup (default: "binance").
        data: Pre-loaded data dict mapping "ASSET/interval" to DataFrames.
        mute_strategy_prints: Suppress print output from strategy code.

    Returns:
        Dict with trades, metrics, simulation_info, and final_balances.
    """
    engine = BacktestEngine(
        interval=interval,
        initial_balances=initial_balances,
        start_time=start_time,
        end_time=end_time,
        exchange=exchange,
        data=data,
        mute_strategy_prints=mute_strategy_prints,
    )
    return engine.run(strategy_code)


__all__ = [
    "BacktestEngine",
    "StaticSandbox",
    "MetricsCalculator",
    "run",
]
