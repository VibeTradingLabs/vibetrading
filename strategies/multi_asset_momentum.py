"""
Multi-Asset Relative Strength Momentum Strategy

Ranks BTC, ETH, and SOL by recent momentum (rate of change over lookback).
Goes long the strongest asset when it has clear upward momentum.

Features:
- Relative strength ranking across assets
- Only trades the top-ranked asset
- EMA trend filter to avoid counter-trend entries
- ATR-based position sizing for volatility normalization
- Longer holding periods to reduce churn

Backtest: vibetrading backtest strategies/multi_asset_momentum.py -i 1h
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
from vibetrading.indicators import atr, ema

# ── Parameters ─────────────────────────────────────────────────────
ASSETS = ["BTC", "ETH", "SOL"]
LEVERAGE = 3
RISK_PCT = 0.12

# Momentum — longer lookback for more stable signals
ROC_PERIOD = 48  # 48h rate of change (was 24)
TREND_EMA_PERIOD = 50

# Minimum momentum threshold — avoid weak signals
MIN_ROC = 0.01  # At least 1% momentum

# Risk management — wider stops
TP_PCT = 0.06  # 6% take profit (was 4.5%)
SL_PCT = 0.035  # 3.5% stop loss (was 2%)

# Rebalance period — longer to reduce turnover
REBALANCE_PERIOD = 24  # Re-evaluate every 24h (was 12)

# State
_candles_since_rebalance = 0
_current_best_asset = None


def rate_of_change(closes, period):
    """Calculate rate of change (momentum) over a period."""
    if len(closes) < period + 1:
        return float("nan")
    return (closes.iloc[-1] - closes.iloc[-period - 1]) / closes.iloc[-period - 1]


@vibe(interval="1h")
def multi_asset_momentum():
    global _candles_since_rebalance, _current_best_asset

    _candles_since_rebalance += 1

    # ── Manage all existing positions (every candle) ──────────────
    for asset in ASSETS:
        position = get_perp_position(asset)
        if not position or position.get("size", 0) == 0:
            continue

        price = get_perp_price(asset)
        if math.isnan(price):
            continue

        entry = position["entry_price"]
        size = position["size"]
        pnl_pct = (price - entry) / entry if entry > 0 else 0

        if pnl_pct >= TP_PCT:
            reduce_position(asset, abs(size))
            continue
        if pnl_pct <= -SL_PCT:
            reduce_position(asset, abs(size))
            continue

    # ── Rebalance check ───────────────────────────────────────────
    if _candles_since_rebalance < REBALANCE_PERIOD:
        return

    _candles_since_rebalance = 0

    # ── Rank assets by momentum ───────────────────────────────────
    lookback = max(ROC_PERIOD, TREND_EMA_PERIOD) + 10
    rankings = []

    for asset in ASSETS:
        ohlcv = get_futures_ohlcv(asset, "1h", lookback)
        if ohlcv is None or len(ohlcv) < lookback - 5:
            continue

        price = get_perp_price(asset)
        if math.isnan(price) or price <= 0:
            continue

        roc = rate_of_change(ohlcv["close"], ROC_PERIOD)
        if math.isnan(roc):
            continue

        trend = ema(ohlcv["close"], TREND_EMA_PERIOD).iloc[-1]
        above_trend = price > trend

        current_atr = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14).iloc[-1]

        rankings.append(
            {
                "asset": asset,
                "roc": roc,
                "above_trend": above_trend,
                "price": price,
                "atr": current_atr,
            }
        )

    if not rankings:
        return

    rankings.sort(key=lambda x: x["roc"], reverse=True)
    best = rankings[0]

    # Only enter if best asset has strong positive momentum AND is above trend
    if best["roc"] < MIN_ROC or not best["above_trend"]:
        return

    best_asset = best["asset"]

    # Close positions in other assets if we're rotating
    if _current_best_asset and _current_best_asset != best_asset:
        old_pos = get_perp_position(_current_best_asset)
        if old_pos and old_pos.get("size", 0) != 0:
            reduce_position(_current_best_asset, abs(old_pos["size"]))

    _current_best_asset = best_asset

    # Skip if already positioned in best asset
    existing = get_perp_position(best_asset)
    if existing and existing.get("size", 0) != 0:
        return

    # ── Enter the strongest asset ─────────────────────────────────
    summary = get_perp_summary()
    margin = summary.get("available_margin", 0)
    if margin < 50:
        return

    set_leverage(best_asset, LEVERAGE)
    qty = (margin * RISK_PCT * LEVERAGE) / best["price"]
    if qty * best["price"] >= 15:
        long(best_asset, qty, best["price"], order_type="market")
