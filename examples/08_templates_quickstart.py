"""
Example 08: Strategy Templates Quick Start

Generate and backtest strategies using built-in templates — no LLM needed.
"""

from datetime import datetime, timezone

import vibetrading.backtest
import vibetrading.tools
from vibetrading.templates import momentum, mean_reversion, dca


def main():
    # ── Download data ──────────────────────────────────────────────
    print("Downloading BTC data...")
    data = vibetrading.tools.download_data(
        ["BTC"],
        exchange="binance",
        interval="1h",
    )

    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 4, 1, tzinfo=timezone.utc)

    # ── Momentum Strategy ──────────────────────────────────────────
    print("\n=== Momentum Strategy ===")
    code = momentum.generate(
        asset="BTC",
        leverage=3,
        sma_fast=10,
        sma_slow=30,
        rsi_period=14,
        tp_pct=0.05,
        sl_pct=0.025,
    )

    result = vibetrading.backtest.run(
        code,
        interval="1h",
        data=data,
        initial_balances={"USDC": 10000},
        start_time=start,
        end_time=end,
        slippage_bps=5,
    )
    _print_summary(result)

    # ── Mean Reversion Strategy ────────────────────────────────────
    print("\n=== Mean Reversion Strategy ===")
    code = mean_reversion.generate(
        asset="BTC",
        leverage=3,
        bb_period=20,
        bb_std=2.0,
        rsi_entry=30,
        tp_pct=0.03,
        sl_pct=0.02,
    )

    result = vibetrading.backtest.run(
        code,
        interval="1h",
        data=data,
        initial_balances={"USDC": 10000},
        start_time=start,
        end_time=end,
        slippage_bps=5,
    )
    _print_summary(result)

    # ── DCA Strategy ───────────────────────────────────────────────
    print("\n=== DCA Strategy ===")
    code = dca.generate(
        asset="BTC",
        buy_amount=50,
        tp_pct=0.15,
        interval="1d",
    )

    result = vibetrading.backtest.run(
        code,
        interval="1d",
        data=data,
        initial_balances={"USDC": 10000},
        start_time=start,
        end_time=end,
    )
    _print_summary(result)


def _print_summary(result):
    if result is None:
        print("  No results")
        return

    m = result["metrics"]
    print(f"  Return:        {m.get('total_return', 0):.2%}")
    print(f"  Sharpe:        {m.get('sharpe_ratio', 0):.3f}")
    print(f"  Sortino:       {m.get('sortino_ratio', 0):.3f}")
    print(f"  Max Drawdown:  {m.get('max_drawdown', 0):.2%}")
    print(f"  Win Rate:      {m.get('win_rate', 0):.2%}")
    print(f"  Profit Factor: {m.get('profit_factor', 0):.2f}")
    print(f"  Trades:        {m.get('number_of_trades', 0)}")


if __name__ == "__main__":
    main()
