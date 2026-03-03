"""
MACD Trend Following Strategy

Uses MACD crossover for trend direction and ATR for dynamic stop-loss
placement. Enters on MACD signal cross when the histogram is building
momentum, and trails the stop using ATR.

Features:
- ATR-based dynamic stop loss (adapts to volatility)
- Partial profit taking at 2x ATR
- Trailing stop after initial TP hit

Backtest: vibetrading backtest strategies/macd_trend_follower.py -i 1h
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
    short,
    vibe,
)
from vibetrading.indicators import atr, ema, macd

# ── Parameters ─────────────────────────────────────────────────────
ASSET = "BTC"
LEVERAGE = 3
RISK_PCT = 0.10

# MACD
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# ATR for stops
ATR_PERIOD = 14
ATR_SL_MULT = 2.0  # Stop loss at 2x ATR from entry
ATR_TP_MULT = 4.0  # Take profit at 4x ATR from entry

# Trend filter: price must be above/below this EMA
TREND_EMA_PERIOD = 50

# State
_partial_taken = False


@vibe(interval="1h")
def macd_trend_follower():
    global _partial_taken

    price = get_perp_price(ASSET)
    if math.isnan(price) or price <= 0:
        return

    # Get data for all indicators
    lookback = max(MACD_SLOW, TREND_EMA_PERIOD, ATR_PERIOD) + 15
    ohlcv = get_futures_ohlcv(ASSET, "1h", lookback)
    if ohlcv is None or len(ohlcv) < lookback - 5:
        return

    # Calculate indicators
    macd_line, signal_line, histogram = macd(ohlcv["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    current_atr = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], ATR_PERIOD).iloc[-1]
    trend_ema = ema(ohlcv["close"], TREND_EMA_PERIOD).iloc[-1]

    if math.isnan(current_atr) or math.isnan(trend_ema):
        return

    # ── Manage existing position ──────────────────────────────────
    position = get_perp_position(ASSET)
    if position and position.get("size", 0) != 0:
        entry = position["entry_price"]
        size = position["size"]
        is_long = size > 0
        abs_size = abs(size)

        if is_long:
            pnl_distance = price - entry
        else:
            pnl_distance = entry - price

        # Dynamic stop loss based on ATR
        if pnl_distance <= -current_atr * ATR_SL_MULT:
            reduce_position(ASSET, abs_size)
            _partial_taken = False
            return

        # Partial take profit at 2x ATR
        if not _partial_taken and pnl_distance >= current_atr * (ATR_TP_MULT / 2):
            reduce_position(ASSET, abs_size * 0.5)
            _partial_taken = True
            return

        # Full take profit at 4x ATR
        if pnl_distance >= current_atr * ATR_TP_MULT:
            reduce_position(ASSET, abs_size)
            _partial_taken = False
            return

        # MACD reversal exit: if histogram flips against us
        if is_long and histogram.iloc[-1] < 0 and histogram.iloc[-2] >= 0:
            reduce_position(ASSET, abs_size)
            _partial_taken = False
            return
        if not is_long and histogram.iloc[-1] > 0 and histogram.iloc[-2] <= 0:
            reduce_position(ASSET, abs_size)
            _partial_taken = False
            return

        return

    _partial_taken = False

    # ── Entry logic ───────────────────────────────────────────────

    # MACD crossover detection
    macd_cross_up = macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]
    macd_cross_down = macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]

    # Histogram momentum confirmation
    hist_building_up = histogram.iloc[-1] > histogram.iloc[-2] > 0
    hist_building_down = histogram.iloc[-1] < histogram.iloc[-2] < 0

    summary = get_perp_summary()
    margin = summary.get("available_margin", 0)
    if margin < 50:
        return

    set_leverage(ASSET, LEVERAGE)

    # Long entry: MACD cross up + price above trend EMA
    if (macd_cross_up or hist_building_up) and price > trend_ema:
        qty = (margin * RISK_PCT * LEVERAGE) / price
        if qty * price >= 15:
            long(ASSET, qty, price, order_type="market")
        return

    # Short entry: MACD cross down + price below trend EMA
    if (macd_cross_down or hist_building_down) and price < trend_ema:
        qty = (margin * RISK_PCT * LEVERAGE) / price
        if qty * price >= 15:
            short(ASSET, qty, price, order_type="market")
        return
