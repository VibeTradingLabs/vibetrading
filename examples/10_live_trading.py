"""
Example 10: Live Trading on Hyperliquid

Run a strategy against a real exchange. This example uses Hyperliquid
but works with any supported exchange (paradex, lighter, aster).

Setup:
    1. pip install "vibetrading[hyperliquid]"
    2. Create .env.local with your credentials:
        HYPERLIQUID_WALLET=0xYourWalletAddress
        HYPERLIQUID_PRIVATE_KEY=0xYourPrivateKey
    3. python examples/10_live_trading.py

⚠️  This trades with real money. Start with small amounts.
"""

import asyncio
import os

# Load credentials from .env.local
try:
    from dotenv import load_dotenv

    load_dotenv(".env.local")
except ImportError:
    pass  # dotenv is optional — you can set env vars directly

# ── Strategy code ─────────────────────────────────────────────────
# This is the exact same code you'd use in a backtest.
# The sandbox handles the difference between simulation and live.

STRATEGY = """
import math
from vibetrading import (
    vibe, get_perp_price, get_perp_position, get_perp_summary,
    get_futures_ohlcv, set_leverage, long, reduce_position,
)
from vibetrading.indicators import rsi

@vibe(interval="1h")
def rsi_strategy():
    price = get_perp_price("BTC")
    if math.isnan(price) or price <= 0:
        return

    # ── Manage position ───────────────────────────────────────────
    position = get_perp_position("BTC")
    if position and position.get("size", 0) != 0:
        entry = position["entry_price"]
        pnl_pct = (price - entry) / entry if entry > 0 else 0

        # Take profit at 3%
        if pnl_pct >= 0.03:
            reduce_position("BTC", abs(position["size"]))
            return

        # Stop loss at 2%
        if pnl_pct <= -0.02:
            reduce_position("BTC", abs(position["size"]))
            return

        return

    # ── Entry: RSI oversold ───────────────────────────────────────
    ohlcv = get_futures_ohlcv("BTC", "1h", 20)
    if ohlcv is None or len(ohlcv) < 15:
        return

    current_rsi = rsi(ohlcv["close"], 14).iloc[-1]
    if math.isnan(current_rsi):
        return

    if current_rsi < 30:
        summary = get_perp_summary()
        margin = summary.get("available_margin", 0)
        if margin > 100:
            set_leverage("BTC", 3)
            qty = (margin * 0.10 * 3) / price
            if qty * price >= 15:
                long("BTC", qty, price, order_type="market")
"""


async def main():
    wallet = os.environ.get("HYPERLIQUID_WALLET")
    private_key = os.environ.get("HYPERLIQUID_PRIVATE_KEY")

    if not wallet or not private_key:
        print("Missing credentials. Set HYPERLIQUID_WALLET and HYPERLIQUID_PRIVATE_KEY")
        print("in .env.local or as environment variables.")
        return

    import vibetrading.live

    print("Starting live trading on Hyperliquid...")
    print("Press Ctrl+C to stop.\n")

    await vibetrading.live.start(
        STRATEGY,
        exchange="hyperliquid",
        api_key=wallet,
        api_secret=private_key,
        interval="1m",  # Check every minute
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
