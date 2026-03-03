"""
Funding Rate Arbitrage Strategy

Captures funding rate payments by positioning against the consensus.
When funding rate is highly positive (longs pay shorts), opens a short
to collect funding. When highly negative, opens a long.

Risk management:
- Only enters when funding rate exceeds a threshold (avoids noise)
- Uses tight stops since this is a carry trade, not a directional bet
- Exits when funding normalizes (carry opportunity ends)

Backtest: vibetrading backtest strategies/funding_rate_arb.py -i 1h
"""

import math

from vibetrading import (
    get_funding_rate,
    get_funding_rate_history,
    get_perp_position,
    get_perp_price,
    get_perp_summary,
    long,
    reduce_position,
    set_leverage,
    short,
    vibe,
)

# ── Parameters ─────────────────────────────────────────────────────
ASSET = "BTC"
LEVERAGE = 2  # Low leverage — this is a carry trade
RISK_PCT = 0.15

# Funding thresholds (annualized)
# Typical BTC funding: ~0.01% per 8h = ~0.03% per day
# We enter when it's significantly elevated
FUNDING_ENTRY_THRESHOLD = 0.0003  # 0.03% per period (high)
FUNDING_EXIT_THRESHOLD = 0.0001  # 0.01% per period (normalized)

# Risk management
SL_PCT = 0.015  # 1.5% stop — tight since we're not betting on direction
MAX_HOLD_PERIODS = 72  # Max 72 candles (~3 days on 1h) before forced exit

# State
_hold_periods = 0


@vibe(interval="1h")
def funding_rate_arb():
    global _hold_periods

    price = get_perp_price(ASSET)
    if math.isnan(price) or price <= 0:
        return

    funding_rate = get_funding_rate(ASSET)
    if math.isnan(funding_rate):
        return

    # ── Manage existing position ──────────────────────────────────
    position = get_perp_position(ASSET)
    if position and position.get("size", 0) != 0:
        entry = position["entry_price"]
        size = position["size"]
        is_long = size > 0
        abs_size = abs(size)
        _hold_periods += 1

        # PnL check
        if is_long:
            pnl_pct = (price - entry) / entry if entry > 0 else 0
        else:
            pnl_pct = (entry - price) / entry if entry > 0 else 0

        # Stop loss — tight since this isn't directional
        if pnl_pct <= -SL_PCT:
            reduce_position(ASSET, abs_size)
            _hold_periods = 0
            return

        # Max hold time exceeded
        if _hold_periods >= MAX_HOLD_PERIODS:
            reduce_position(ASSET, abs_size)
            _hold_periods = 0
            return

        # Funding normalized — carry opportunity over
        if abs(funding_rate) < FUNDING_EXIT_THRESHOLD:
            reduce_position(ASSET, abs_size)
            _hold_periods = 0
            return

        # Funding flipped against us
        if is_long and funding_rate > FUNDING_ENTRY_THRESHOLD:
            # We're long but longs are paying — wrong side
            reduce_position(ASSET, abs_size)
            _hold_periods = 0
            return
        if not is_long and funding_rate < -FUNDING_ENTRY_THRESHOLD:
            # We're short but shorts are paying — wrong side
            reduce_position(ASSET, abs_size)
            _hold_periods = 0
            return

        return

    _hold_periods = 0

    # ── Entry logic ───────────────────────────────────────────────

    # Check funding rate history for persistence (not just a spike)
    fr_history = get_funding_rate_history(ASSET, 8)
    if fr_history is None or len(fr_history) < 3:
        return

    recent_rates = fr_history["fundingRate"].tail(3)
    avg_recent = recent_rates.mean()

    summary = get_perp_summary()
    margin = summary.get("available_margin", 0)
    if margin < 50:
        return

    set_leverage(ASSET, LEVERAGE)
    qty = (margin * RISK_PCT * LEVERAGE) / price
    if qty * price < 15:
        return

    # High positive funding → go short (collect from longs)
    if avg_recent > FUNDING_ENTRY_THRESHOLD and funding_rate > FUNDING_ENTRY_THRESHOLD:
        short(ASSET, qty, price, order_type="market")
        return

    # High negative funding → go long (collect from shorts)
    if avg_recent < -FUNDING_ENTRY_THRESHOLD and funding_rate < -FUNDING_ENTRY_THRESHOLD:
        long(ASSET, qty, price, order_type="market")
        return
