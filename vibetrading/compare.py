"""
Strategy comparison — run and compare multiple strategies side by side.

Usage::

    import vibetrading.compare

    results = vibetrading.compare.run(
        {
            "RSI Mean Reversion": open("strategies/rsi_mean_reversion.py").read(),
            "MACD Trend": open("strategies/macd_trend_follower.py").read(),
            "DCA": open("strategies/spot_dca_rebalance.py").read(),
        },
        interval="1h",
        initial_balances={"USDC": 10000},
        slippage_bps=5,
    )

    # Print comparison table
    vibetrading.compare.print_table(results)

    # Get as DataFrame
    df = vibetrading.compare.to_dataframe(results)
"""

from datetime import datetime
from typing import Any

from .backtest import run as backtest_run


def run(
    strategies: dict[str, str],
    *,
    interval: str = "1h",
    initial_balances: dict[str, float] | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    exchange: str = "binance",
    data: dict | None = None,
    mute_strategy_prints: bool = True,
    slippage_bps: float = 0.0,
) -> dict[str, dict[str, Any] | None]:
    """
    Run multiple strategies and collect results for comparison.

    Args:
        strategies: Dict mapping strategy name to strategy code string.
        interval: Candle interval (e.g. "1h", "4h", "1d").
        initial_balances: Starting balances (default: {"USDC": 10000}).
        start_time: Backtest start time.
        end_time: Backtest end time.
        exchange: Exchange for data download.
        data: Pre-loaded data dict.
        mute_strategy_prints: Suppress strategy print output (default: True).
        slippage_bps: Slippage in basis points.

    Returns:
        Dict mapping strategy name to backtest result dict (or None on error).
    """
    results: dict[str, dict[str, Any] | None] = {}

    for name, code in strategies.items():
        try:
            result = backtest_run(
                code,
                interval=interval,
                initial_balances=initial_balances,
                start_time=start_time,
                end_time=end_time,
                exchange=exchange,
                data=data,
                mute_strategy_prints=mute_strategy_prints,
                slippage_bps=slippage_bps,
            )
            results[name] = result
        except Exception as e:
            print(f"  {name}: ERROR — {type(e).__name__}: {e}")
            results[name] = None

    return results


def print_table(results: dict[str, dict[str, Any] | None]) -> None:
    """
    Print a formatted comparison table to stdout.

    Args:
        results: Output from compare.run().
    """
    header = f"{'Strategy':<25} {'Return':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'WinRate':>8} {'PF':>6} {'Trades':>7} {'Final':>12}"
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    for name, result in results.items():
        if result is None:
            print(f"{name:<25} {'ERROR':>8}")
            continue

        m = result.get("metrics", {})
        print(
            f"{name:<25} "
            f"{m.get('total_return', 0):>+7.2%} "
            f"{m.get('sharpe_ratio', 0):>8.3f} "
            f"{m.get('sortino_ratio', 0):>8.3f} "
            f"{m.get('max_drawdown', 0):>7.2%} "
            f"{m.get('win_rate', 0):>7.2%} "
            f"{m.get('profit_factor', 0):>6.2f} "
            f"{m.get('number_of_trades', 0):>7} "
            f"${m.get('total_value', 0):>10,.2f}"
        )

    print(sep)

    # Find best strategy by Sharpe
    valid = {k: v for k, v in results.items() if v is not None}
    if valid:
        best_sharpe = max(valid, key=lambda k: valid[k].get("metrics", {}).get("sharpe_ratio", float("-inf")))
        best_return = max(valid, key=lambda k: valid[k].get("metrics", {}).get("total_return", float("-inf")))
        print(f"Best Sharpe: {best_sharpe}")
        print(f"Best Return: {best_return}")


def to_dataframe(results: dict[str, dict[str, Any] | None]) -> Any:
    """
    Convert comparison results to a pandas DataFrame.

    Args:
        results: Output from compare.run().

    Returns:
        pandas DataFrame with one row per strategy and metrics as columns.
    """
    import pandas as pd

    rows = []
    for name, result in results.items():
        if result is None:
            rows.append({"strategy": name, "error": True})
            continue

        m = result.get("metrics", {})
        rows.append(
            {
                "strategy": name,
                "error": False,
                "total_return": m.get("total_return", 0),
                "cagr": m.get("cagr", 0),
                "sharpe_ratio": m.get("sharpe_ratio", 0),
                "sortino_ratio": m.get("sortino_ratio", 0),
                "calmar_ratio": m.get("calmar_ratio", 0),
                "max_drawdown": m.get("max_drawdown", 0),
                "max_drawdown_duration_hours": m.get("max_drawdown_duration_hours", 0),
                "win_rate": m.get("win_rate", 0),
                "profit_factor": m.get("profit_factor", 0),
                "expectancy": m.get("expectancy", 0),
                "number_of_trades": m.get("number_of_trades", 0),
                "winning_trades": m.get("winning_trades", 0),
                "losing_trades": m.get("losing_trades", 0),
                "avg_win": m.get("avg_win", 0),
                "avg_loss": m.get("avg_loss", 0),
                "largest_win": m.get("largest_win", 0),
                "largest_loss": m.get("largest_loss", 0),
                "total_tx_fees": m.get("total_tx_fees", 0),
                "funding_revenue": m.get("funding_revenue", 0),
                "total_value": m.get("total_value", 0),
            }
        )

    return pd.DataFrame(rows).set_index("strategy")


__all__ = [
    "run",
    "print_table",
    "to_dataframe",
]
