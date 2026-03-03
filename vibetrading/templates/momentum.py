"""
Momentum Strategy Template

Opens a long position when short-term SMA crosses above long-term SMA,
and closes when it crosses below. Uses RSI for confirmation.

Parameters (customize via the constants at the top):
- ASSET: Trading pair (default: BTC)
- LEVERAGE: Position leverage (default: 3)
- RISK_PCT: Fraction of available margin per trade (default: 0.15)
- SMA_FAST / SMA_SLOW: Moving average periods
- RSI_PERIOD / RSI_OVERSOLD / RSI_OVERBOUGHT: RSI filter
- TP_PCT / SL_PCT: Take-profit and stop-loss thresholds
"""

TEMPLATE = '''
import math
from vibetrading import (
    vibe, get_perp_price, get_futures_ohlcv, get_perp_summary,
    get_perp_position, set_leverage, long, reduce_position,
)

# ── Parameters ─────────────────────────────────────────────────────
ASSET = "{asset}"
LEVERAGE = {leverage}
RISK_PCT = {risk_pct}
SMA_FAST = {sma_fast}
SMA_SLOW = {sma_slow}
RSI_PERIOD = {rsi_period}
RSI_OVERSOLD = {rsi_oversold}
RSI_OVERBOUGHT = {rsi_overbought}
TP_PCT = {tp_pct}
SL_PCT = {sl_pct}

def compute_rsi(closes, period):
    """Compute RSI from a pandas Series of close prices."""
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("inf"))
    return 100 - (100 / (1 + rs))

@vibe(interval="{interval}")
def momentum_strategy():
    price = get_perp_price(ASSET)
    if math.isnan(price) or price <= 0:
        return

    # Check existing position for TP/SL
    position = get_perp_position(ASSET)
    if position and position.get("size", 0) != 0:
        entry = position["entry_price"]
        size = position["size"]
        pnl_pct = (price - entry) / entry if entry > 0 else 0

        if size > 0:
            if pnl_pct >= TP_PCT:
                reduce_position(ASSET, abs(size))
                return
            if pnl_pct <= -SL_PCT:
                reduce_position(ASSET, abs(size))
                return
        return  # Hold position, no new entries

    # Get OHLCV data
    ohlcv = get_futures_ohlcv(ASSET, "{interval}", max(SMA_SLOW, RSI_PERIOD) + 10)
    if ohlcv is None or len(ohlcv) < SMA_SLOW + 2:
        return

    closes = ohlcv["close"]

    # Moving average crossover
    sma_fast = closes.rolling(SMA_FAST).mean()
    sma_slow = closes.rolling(SMA_SLOW).mean()

    if sma_fast.iloc[-1] <= sma_slow.iloc[-1]:
        return  # No bullish crossover

    if sma_fast.iloc[-2] >= sma_slow.iloc[-2]:
        return  # Not a fresh crossover

    # RSI confirmation
    rsi = compute_rsi(closes, RSI_PERIOD)
    current_rsi = rsi.iloc[-1]
    if math.isnan(current_rsi) or current_rsi > RSI_OVERBOUGHT:
        return  # Overbought, skip

    # Size and enter
    summary = get_perp_summary()
    margin = summary.get("available_margin", 0)
    if margin < 50:
        return

    set_leverage(ASSET, LEVERAGE)
    qty = (margin * RISK_PCT * LEVERAGE) / price
    if qty * price >= 15:
        long(ASSET, qty, price, order_type="market")
'''

DEFAULTS = {
    "asset": "BTC",
    "leverage": 3,
    "risk_pct": 0.15,
    "sma_fast": 10,
    "sma_slow": 30,
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "tp_pct": 0.03,
    "sl_pct": 0.015,
    "interval": "1h",
}


def generate(**kwargs) -> str:
    """Generate momentum strategy code with custom parameters."""
    params = {**DEFAULTS, **kwargs}
    return TEMPLATE.format(**params).strip()
