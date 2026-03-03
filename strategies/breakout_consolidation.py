"""
Volatility Breakout Strategy

Detects consolidation periods (low volatility) and enters on breakouts.
Uses Bollinger Band width to measure volatility compression — when bands
squeeze tight, a breakout is likely. Enters in the direction of the breakout.

Features:
- BB width squeeze detection for consolidation
- Breakout confirmation with volume spike
- ATR-based stop loss (fixed from entry, not trailing while losing)
- Partial profit taking

Backtest: vibetrading backtest strategies/breakout_consolidation.py -i 1h
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
from vibetrading.indicators import atr, bbands

# ── Parameters ─────────────────────────────────────────────────────
ASSET = "BTC"
LEVERAGE = 3
RISK_PCT = 0.12

# Bollinger Bands
BB_PERIOD = 20
BB_STD = 2.0

# Squeeze detection
SQUEEZE_LOOKBACK = 100  # Longer lookback for more reliable percentiles
SQUEEZE_PERCENTILE = 0.15  # Tighter squeeze requirement (15th percentile)

# Volume confirmation
VOLUME_SPIKE_MULT = 1.3  # Slightly relaxed volume requirement

# Risk management
ATR_PERIOD = 14
ATR_SL_MULT = 2.5  # Wider stop to survive initial volatility
TP_PCT = 0.08  # 8% take profit — breakouts should run
PARTIAL_TP_PCT = 0.04  # 4% partial

# State
_in_squeeze = False
_squeeze_candles = 0  # Track how long we've been in squeeze
_partial_taken = False
_entry_atr = 0.0  # Store ATR at entry for fixed stop


@vibe(interval="1h")
def breakout_consolidation():
    global _in_squeeze, _squeeze_candles, _partial_taken, _entry_atr

    price = get_perp_price(ASSET)
    if math.isnan(price) or price <= 0:
        return

    lookback = max(BB_PERIOD, SQUEEZE_LOOKBACK, ATR_PERIOD) + 10
    ohlcv = get_futures_ohlcv(ASSET, "1h", lookback)
    if ohlcv is None or len(ohlcv) < lookback - 5:
        return

    upper, middle, lower = bbands(ohlcv["close"], BB_PERIOD, BB_STD)
    current_atr = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], ATR_PERIOD).iloc[-1]

    if math.isnan(current_atr):
        return

    # ── Manage existing position ──────────────────────────────────
    position = get_perp_position(ASSET)
    if position and position.get("size", 0) != 0:
        entry = position["entry_price"]
        size = position["size"]
        is_long = size > 0
        abs_size = abs(size)

        if is_long:
            pnl_pct = (price - entry) / entry if entry > 0 else 0
        else:
            pnl_pct = (entry - price) / entry if entry > 0 else 0

        # Fixed ATR stop from entry (use entry ATR, not current)
        stop_atr = _entry_atr if _entry_atr > 0 else current_atr
        stop_distance = stop_atr * ATR_SL_MULT / entry if entry > 0 else 0.05
        if pnl_pct <= -stop_distance:
            reduce_position(ASSET, abs_size)
            _partial_taken = False
            _entry_atr = 0.0
            return

        # Partial take profit
        if not _partial_taken and pnl_pct >= PARTIAL_TP_PCT:
            reduce_position(ASSET, abs_size * 0.4)
            _partial_taken = True
            return

        # Full take profit
        if pnl_pct >= TP_PCT:
            reduce_position(ASSET, abs_size)
            _partial_taken = False
            _entry_atr = 0.0
            return

        return

    _partial_taken = False

    # ── Squeeze detection ─────────────────────────────────────────
    bb_width = (upper - lower) / middle
    valid_widths = bb_width.dropna().tail(SQUEEZE_LOOKBACK)

    if len(valid_widths) < SQUEEZE_LOOKBACK // 2:
        return

    current_width = bb_width.iloc[-1]
    if math.isnan(current_width):
        return

    width_percentile = (valid_widths < current_width).sum() / len(valid_widths)

    was_in_squeeze = _in_squeeze
    if width_percentile < SQUEEZE_PERCENTILE:
        _in_squeeze = True
        _squeeze_candles += 1
    else:
        if _in_squeeze:
            # Just exited squeeze — this is the breakout candle
            pass
        _in_squeeze = False
        _squeeze_candles = 0

    # Only look for breakouts right after a squeeze (squeeze must have lasted 3+ candles)
    if not was_in_squeeze:
        return
    if _squeeze_candles < 3:
        pass  # Short squeeze — still check for breakout but less reliable

    # Breakout = price moves outside the bands after squeeze
    breakout_up = price > upper.iloc[-1]
    breakout_down = price < lower.iloc[-1]

    if not breakout_up and not breakout_down:
        return

    # Volume confirmation
    avg_volume = ohlcv["volume"].tail(20).mean()
    current_volume = ohlcv["volume"].iloc[-1]
    if current_volume < avg_volume * VOLUME_SPIKE_MULT:
        return

    # ── Enter ─────────────────────────────────────────────────────
    summary = get_perp_summary()
    margin = summary.get("available_margin", 0)
    if margin < 50:
        return

    set_leverage(ASSET, LEVERAGE)
    qty = (margin * RISK_PCT * LEVERAGE) / price
    if qty * price < 15:
        return

    _entry_atr = current_atr
    _squeeze_candles = 0

    if breakout_up:
        long(ASSET, qty, price, order_type="market")
    elif breakout_down:
        short(ASSET, qty, price, order_type="market")
