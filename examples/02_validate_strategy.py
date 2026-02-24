"""
Example 2: Validate strategy code before running.

Demonstrates using vibetrading.strategy.validate() to catch common issues
in strategy code — missing decorators, leverage not set, hardcoded balances, etc.

Usage:
    python examples/02_validate_strategy.py
"""

import vibetrading.strategy


# ── Good strategy (passes validation) ──────────────────────────────────
good_strategy = """
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


@vibe(interval="1m")
def my_strategy():
    price = get_perp_price(ASSET)
    if math.isnan(price):
        return

    summary = get_perp_summary()
    margin = summary.get("available_margin", 0.0)
    position = get_perp_position(ASSET)

    if position:
        size = position.get("size", 0.0)
        entry = position.get("entry_price", 0.0)
        if entry > 0 and size > 0:
            pnl_pct = (price - entry) / entry
            if pnl_pct >= TP_PCT:
                reduce_position(ASSET, abs(size) * 0.5)
                return
            elif pnl_pct <= -SL_PCT:
                reduce_position(ASSET, abs(size))
                return
        return

    ohlcv = get_futures_ohlcv(ASSET, "1m", 30)
    if len(ohlcv) < 20:
        return

    sma = ohlcv["close"].rolling(20).mean().iloc[-1]
    if price > sma:
        set_leverage(ASSET, LEVERAGE)
        qty = (margin * RISK_PER_TRADE_PCT * LEVERAGE) / price
        if qty * price >= 15.0:
            long(ASSET, qty, price=price)
"""


# ── Bad strategy (multiple issues) ─────────────────────────────────────
bad_strategy = """
from vibetrading import long, short, get_price, get_futures_ohlcv

def trade():
    price = get_price("BTC")
    balance = 10000
    qty = balance * 0.1 / price
    if price > 50000:
        long("BTC", qty, price=price)
    else:
        short("BTC", qty, price=price)
"""


def main():
    print("=" * 60)
    print("Example 2: Strategy Validation")
    print("=" * 60)

    # Validate the good strategy
    print("\n--- Validating good strategy ---\n")
    result = vibetrading.strategy.validate(good_strategy)
    print(f"Valid: {result.is_valid}")
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")
    if result.warnings:
        for w in result.warnings:
            print(f"  WARN: {w}")
    print()

    # Validate the bad strategy
    print("--- Validating bad strategy ---\n")
    result = vibetrading.strategy.validate(bad_strategy)
    print(f"Valid: {result.is_valid}")
    print(f"Errors: {len(result.errors)}")
    for e in result.errors:
        print(f"  ERROR: {e}")
    print(f"Warnings: {len(result.warnings)}")
    for w in result.warnings:
        print(f"  WARN: {w}")
    print()

    # Show LLM feedback format
    print("--- LLM feedback format ---\n")
    feedback = result.format_for_llm()
    print(feedback)


if __name__ == "__main__":
    main()
