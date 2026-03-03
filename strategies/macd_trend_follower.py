"""
MACD Trend Following Strategy

Uses MACD crossover for trend direction and ATR for dynamic stop-loss
placement. Enters only on clean MACD signal line crossovers when confirmed
by the trend EMA filter. Trails stops using ATR.

Features:
- Strict crossover-only entries (no histogram momentum re-entry)
- ATR-based dynamic stop loss (adapts to volatility)
- Partial profit taking at 2x ATR
- Full exit at 4x ATR or MACD reversal

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
RISK_PCT = 0.15  # Larger position, fewer trades

# MACD
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# ATR for stops
ATR_PERIOD = 14
ATR_SL_MULT = 2.5  # Wider stop — reduce whipsaws
ATR_TP_MULT = 5.0  # Wider TP — let winners run

# Trend filter — dual EMA for stronger confirmation
TREND_EMA_FAST = 50
TREND_EMA_SLOW = 100  # Only trade when 50 > 100 (uptrend) or 50 < 100 (downtrend)

# Cooldown — minimum candles between trades to avoid churn
MIN_CANDLES_BETWEEN_TRADES = 12

# State
_partial_taken = False
_candles_since_last_trade = 999


@vibe(interval="1h")
def macd_trend_follower():
    global _partial_taken, _candles_since_last_trade

    _candles_since_last_trade += 1

    price = get_perp_price(ASSET)
    if math.isnan(price) or price <= 0:
        return

    lookback = max(MACD_SLOW, TREND_EMA_SLOW, ATR_PERIOD) + 15
    ohlcv = get_futures_ohlcv(ASSET, "1h", lookback)
    if ohlcv is None or len(ohlcv) < lookback - 5:
        return

    macd_line, signal_line, histogram = macd(ohlcv["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    current_atr = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], ATR_PERIOD).iloc[-1]
    ema_fast = ema(ohlcv["close"], TREND_EMA_FAST).iloc[-1]
    ema_slow = ema(ohlcv["close"], TREND_EMA_SLOW).iloc[-1]

    if math.isnan(current_atr) or math.isnan(ema_fast) or math.isnan(ema_slow):
        return

    # Trend direction from dual EMA
    uptrend = ema_fast > ema_slow
    downtrend = ema_fast < ema_slow

    # ── Manage existing position ──────────────────────────────────
    position = get_perp_position(ASSET)
    if position and position.get("size", 0) != 0:
        entry = position["entry_price"]
        size = position["size"]
        is_long = size > 0
        abs_size = abs(size)

        pnl_distance = (price - entry) if is_long else (entry - price)

        # Stop loss
        if pnl_distance <= -current_atr * ATR_SL_MULT:
            reduce_position(ASSET, abs_size)
            _partial_taken = False
            _candles_since_last_trade = 0
            return

        # Partial TP at halfway point
        if not _partial_taken and pnl_distance >= current_atr * (ATR_TP_MULT / 2):
            reduce_position(ASSET, abs_size * 0.5)
            _partial_taken = True
            return

        # Full TP
        if pnl_distance >= current_atr * ATR_TP_MULT:
            reduce_position(ASSET, abs_size)
            _partial_taken = False
            _candles_since_last_trade = 0
            return

        # MACD reversal — only exit on confirmed cross, not just histogram flip
        if is_long and macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
            reduce_position(ASSET, abs_size)
            _partial_taken = False
            _candles_since_last_trade = 0
            return
        if not is_long and macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
            reduce_position(ASSET, abs_size)
            _partial_taken = False
            _candles_since_last_trade = 0
            return

        return

    _partial_taken = False

    # ── Entry logic (crossover only, no histogram re-entry) ───────
    if _candles_since_last_trade < MIN_CANDLES_BETWEEN_TRADES:
        return

    macd_cross_up = macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]
    macd_cross_down = macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]

    if not macd_cross_up and not macd_cross_down:
        return

    summary = get_perp_summary()
    margin = summary.get("available_margin", 0)
    if margin < 50:
        return

    set_leverage(ASSET, LEVERAGE)

    if macd_cross_up and uptrend:
        qty = (margin * RISK_PCT * LEVERAGE) / price
        if qty * price >= 15:
            long(ASSET, qty, price, order_type="market")
            _candles_since_last_trade = 0
        return

    if macd_cross_down and downtrend:
        qty = (margin * RISK_PCT * LEVERAGE) / price
        if qty * price >= 15:
            short(ASSET, qty, price, order_type="market")
            _candles_since_last_trade = 0
        return
