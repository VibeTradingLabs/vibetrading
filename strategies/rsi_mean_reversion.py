"""
RSI Mean Reversion Strategy

Enters long when RSI is oversold AND price is below lower Bollinger Band.
Exits at the SMA (mean reversion target) or on TP/SL.

This double-confirmation approach filters out false oversold signals
in trending markets.

Backtest: vibetrading backtest strategies/rsi_mean_reversion.py -i 1h
"""

import math

from vibetrading import (
    get_futures_ohlcv,
    get_perp_position,
    get_perp_price,
    get_perp_summary,
    long,
    reduce_position,
    set_leverage,
    vibe,
)
from vibetrading.indicators import bbands, rsi

# ── Parameters ─────────────────────────────────────────────────────
ASSET = "BTC"
LEVERAGE = 3
RISK_PCT = 0.12  # 12% of available margin per trade

# Indicators
RSI_PERIOD = 14
RSI_OVERSOLD = 28
BB_PERIOD = 20
BB_STD = 2.0

# Risk management
TP_PCT = 0.035  # 3.5% take profit
SL_PCT = 0.018  # 1.8% stop loss

# Cooldown: skip N candles after a stop-loss
SL_COOLDOWN = 5
_cooldown_remaining = 0


@vibe(interval="1h")
def rsi_mean_reversion():
    global _cooldown_remaining

    price = get_perp_price(ASSET)
    if math.isnan(price) or price <= 0:
        return

    # ── Manage existing position ──────────────────────────────────
    position = get_perp_position(ASSET)
    if position and position.get("size", 0) != 0:
        entry = position["entry_price"]
        size = position["size"]
        pnl_pct = (price - entry) / entry if entry > 0 else 0

        # Take profit
        if pnl_pct >= TP_PCT:
            reduce_position(ASSET, abs(size))
            return

        # Stop loss
        if pnl_pct <= -SL_PCT:
            reduce_position(ASSET, abs(size))
            _cooldown_remaining = SL_COOLDOWN
            return

        # Mean reversion exit: price returns to SMA
        ohlcv = get_futures_ohlcv(ASSET, "1h", BB_PERIOD + 5)
        if ohlcv is not None and len(ohlcv) >= BB_PERIOD:
            _, middle, _ = bbands(ohlcv["close"], BB_PERIOD, BB_STD)
            sma_val = middle.iloc[-1]
            if not math.isnan(sma_val) and price >= sma_val and pnl_pct > 0:
                reduce_position(ASSET, abs(size))
        return

    # ── Cooldown after stop loss ──────────────────────────────────
    if _cooldown_remaining > 0:
        _cooldown_remaining -= 1
        return

    # ── Entry logic ───────────────────────────────────────────────
    ohlcv = get_futures_ohlcv(ASSET, "1h", max(RSI_PERIOD, BB_PERIOD) + 10)
    if ohlcv is None or len(ohlcv) < BB_PERIOD + 2:
        return

    closes = ohlcv["close"]

    # RSI check
    rsi_values = rsi(closes, RSI_PERIOD)
    current_rsi = rsi_values.iloc[-1]
    if math.isnan(current_rsi) or current_rsi > RSI_OVERSOLD:
        return

    # Bollinger Band check — price must be below lower band
    upper, middle, lower = bbands(closes, BB_PERIOD, BB_STD)
    if price >= lower.iloc[-1]:
        return

    # ── Size and enter ────────────────────────────────────────────
    summary = get_perp_summary()
    margin = summary.get("available_margin", 0)
    if margin < 50:
        return

    set_leverage(ASSET, LEVERAGE)
    qty = (margin * RISK_PCT * LEVERAGE) / price
    if qty * price >= 15:
        long(ASSET, qty, price, order_type="market")
