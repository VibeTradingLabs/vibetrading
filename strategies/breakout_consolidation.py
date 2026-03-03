"""
Volatility Breakout Strategy

Detects consolidation periods (low volatility) and enters on breakouts.
Uses Bollinger Band width to measure volatility compression — when bands
squeeze tight, a breakout is likely. Enters in the direction of the breakout.

Features:
- BB width squeeze detection for consolidation
- Breakout confirmation with volume spike
- ATR-based trailing stop
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
RISK_PCT = 0.10

# Bollinger Bands
BB_PERIOD = 20
BB_STD = 2.0

# Squeeze detection
# BB width percentile — squeeze when width is below this percentile
SQUEEZE_LOOKBACK = 50  # Candles to calculate width percentile
SQUEEZE_PERCENTILE = 0.20  # Below 20th percentile = squeeze

# Volume confirmation
VOLUME_SPIKE_MULT = 1.5  # Volume must be 1.5x average

# Risk management
ATR_PERIOD = 14
ATR_SL_MULT = 2.0
TP_PCT = 0.06  # 6% take profit
PARTIAL_TP_PCT = 0.03  # 3% partial

# State
_in_squeeze = False
_partial_taken = False


@vibe(interval="1h")
def breakout_consolidation():
    global _in_squeeze, _partial_taken

    price = get_perp_price(ASSET)
    if math.isnan(price) or price <= 0:
        return

    lookback = max(BB_PERIOD, SQUEEZE_LOOKBACK, ATR_PERIOD) + 10
    ohlcv = get_futures_ohlcv(ASSET, "1h", lookback)
    if ohlcv is None or len(ohlcv) < lookback - 5:
        return

    # Calculate indicators
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

        # ATR trailing stop
        if is_long:
            trailing_stop = price - current_atr * ATR_SL_MULT
            if price <= trailing_stop and pnl_pct < 0:
                reduce_position(ASSET, abs_size)
                _partial_taken = False
                return
        else:
            trailing_stop = price + current_atr * ATR_SL_MULT
            if price >= trailing_stop and pnl_pct < 0:
                reduce_position(ASSET, abs_size)
                _partial_taken = False
                return

        # Fixed stop loss
        if pnl_pct <= -(current_atr * ATR_SL_MULT / entry):
            reduce_position(ASSET, abs_size)
            _partial_taken = False
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
            return

        return

    _partial_taken = False

    # ── Squeeze detection ─────────────────────────────────────────

    # Calculate BB width over lookback
    bb_width = (upper - lower) / middle
    valid_widths = bb_width.dropna().tail(SQUEEZE_LOOKBACK)

    if len(valid_widths) < SQUEEZE_LOOKBACK // 2:
        return

    current_width = bb_width.iloc[-1]
    if math.isnan(current_width):
        return

    # Percentile rank of current width
    width_percentile = (valid_widths < current_width).sum() / len(valid_widths)

    # Track squeeze state
    was_in_squeeze = _in_squeeze
    _in_squeeze = width_percentile < SQUEEZE_PERCENTILE

    # ── Breakout detection ────────────────────────────────────────

    # Only look for breakouts after a squeeze period
    if not was_in_squeeze:
        return

    # Breakout = price moves outside the bands after squeeze
    breakout_up = price > upper.iloc[-1]
    breakout_down = price < lower.iloc[-1]

    if not breakout_up and not breakout_down:
        return

    # Volume confirmation
    avg_volume = ohlcv["volume"].tail(20).mean()
    current_volume = ohlcv["volume"].iloc[-1]
    if current_volume < avg_volume * VOLUME_SPIKE_MULT:
        return  # No volume confirmation

    # ── Enter ─────────────────────────────────────────────────────
    summary = get_perp_summary()
    margin = summary.get("available_margin", 0)
    if margin < 50:
        return

    set_leverage(ASSET, LEVERAGE)
    qty = (margin * RISK_PCT * LEVERAGE) / price
    if qty * price < 15:
        return

    if breakout_up:
        long(ASSET, qty, price, order_type="market")
    elif breakout_down:
        short(ASSET, qty, price, order_type="market")
