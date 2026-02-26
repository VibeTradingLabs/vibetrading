"""
Example 1: Backtest a manually written SMA crossover strategy.

Demonstrates the explicit download -> backtest workflow:
  1. vibetrading.tools.download_data() fetches historical OHLCV from Binance
  2. vibetrading.backtest.BacktestEngine receives the data and runs the strategy

Usage:
    python examples/01_backtest_sma_crossover.py
"""
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

import vibetrading.backtest
import vibetrading.tools

load_dotenv()

strategy_code = """
import math
from vibetrading import (
    vibe,
    get_perp_price,
    get_futures_ohlcv,
    get_perp_summary,
    get_perp_position,
    long,
    reduce_position,
    set_leverage,
)

ASSET = "BTC"
LEVERAGE = 3
TP_PCT = 0.08
SL_PCT = 0.04
RISK_PER_TRADE_PCT = 0.10
SMA_FAST = 10
SMA_SLOW = 20
MIN_ORDER_VALUE_USD = 15.0


@vibe(interval="1h")
def sma_crossover():
    current_price = get_perp_price(ASSET)
    if math.isnan(current_price):
        return

    perp_summary = get_perp_summary()
    available_margin = perp_summary.get("available_margin", 0.0)
    position = get_perp_position(ASSET)

    # Risk management (every tick)
    if position:
        size = position.get("size", 0.0)
        entry_price = position.get("entry_price", 0.0)
        if entry_price > 0 and size > 0:
            pnl_pct = (current_price - entry_price) / entry_price
            if pnl_pct >= TP_PCT:
                reduce_position(ASSET, abs(size) * 0.5)
                print(f"TP hit: {pnl_pct:.2%}")
                return
            elif pnl_pct <= -SL_PCT:
                reduce_position(ASSET, abs(size))
                print(f"SL hit: {pnl_pct:.2%}")
                return
        return

    # Entry logic (only when flat)
    ohlcv = get_futures_ohlcv(ASSET, "1h", SMA_SLOW + 10)
    if len(ohlcv) < SMA_SLOW:
        return

    sma_fast = ohlcv["close"].rolling(SMA_FAST).mean().iloc[-1]
    sma_slow = ohlcv["close"].rolling(SMA_SLOW).mean().iloc[-1]

    if sma_fast > sma_slow:
        set_leverage(ASSET, LEVERAGE)
        qty = (available_margin * RISK_PER_TRADE_PCT * LEVERAGE) / current_price
        order_value = qty * current_price
        if order_value >= MIN_ORDER_VALUE_USD:
            result = long(ASSET, qty, price=current_price)
            if result.get("status") == "success":
                print(f"Long entry: SMA {sma_fast:.0f} > {sma_slow:.0f}")
"""


def main():
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 6, 1, tzinfo=timezone.utc)

    # Step 1: Download data explicitly
    print("Step 1: Downloading historical data from Binance...\n")
    data = vibetrading.tools.download_data(
        ["BTC"],
        exchange="binance",
        start_time=start,
        end_time=end,
        interval="1h",
        market_type="perp",
        # proxy=os.environ["HTTPS_PROXY"]
    )
    print(f"\nDownloaded {len(data)} dataset(s).\n")

    # Step 2: Run backtest with pre-downloaded data
    print("Step 2: Running backtest: SMA Crossover on BTC...\n")
    engine = vibetrading.backtest.BacktestEngine(
        start_time=start,
        end_time=end,
        interval="1h",
        exchange="binance",
        initial_balances={"USDC": 10000},
        data=data,
    )

    results = engine.run(strategy_code)

    if results:
        metrics = results["metrics"]
        print("\n=== Backtest Results ===")
        print(f"Total Return:    {metrics['total_return']:.2%}")
        print(f"Max Drawdown:    {metrics['max_drawdown']:.2%}")
        print(f"Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
        print(f"Win Rate:        {metrics['win_rate']:.2%}")
        print(f"Total Trades:    {metrics['number_of_trades']}")
        print(f"Funding Revenue: ${metrics['funding_revenue']:.2f}")
        print(f"Total Fees:      ${metrics['total_tx_fees']:.2f}")
        print(f"Final Value:     ${metrics['total_value']:.2f}")

        info = results["simulation_info"]
        print(f"\nSimulation: {info['time_range']}, {info['steps']} steps")
        if info["liquidated"]:
            print(f"WARNING: Liquidated at {info['liquidation_time']}")
    else:
        print("Backtest returned no results.")


if __name__ == "__main__":
    main()
