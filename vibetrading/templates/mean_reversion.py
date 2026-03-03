"""
Mean Reversion Strategy Template

Enters long when price drops below the lower Bollinger Band and RSI is oversold.
Exits on mean reversion (price returns to SMA) or on stop-loss.

Parameters (customize via the constants at the top):
- ASSET: Trading pair (default: BTC)
- LEVERAGE: Position leverage (default: 3)
- RISK_PCT: Fraction of available margin per trade (default: 0.10)
- BB_PERIOD / BB_STD: Bollinger Band parameters
- RSI_PERIOD / RSI_ENTRY: RSI threshold for entry
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
BB_PERIOD = {bb_period}
BB_STD = {bb_std}
RSI_PERIOD = {rsi_period}
RSI_ENTRY = {rsi_entry}
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
def mean_reversion_strategy():
    price = get_perp_price(ASSET)
    if math.isnan(price) or price <= 0:
        return

    # Check existing position
    position = get_perp_position(ASSET)
    if position and position.get("size", 0) != 0:
        entry = position["entry_price"]
        size = position["size"]
        pnl_pct = (price - entry) / entry if entry > 0 else 0

        if size > 0:
            # Exit at mean (SMA) or TP/SL
            if pnl_pct >= TP_PCT or pnl_pct <= -SL_PCT:
                reduce_position(ASSET, abs(size))
                return

            # Also exit if price returns to SMA (mean reversion target)
            ohlcv = get_futures_ohlcv(ASSET, "{interval}", BB_PERIOD + 5)
            if ohlcv is not None and len(ohlcv) >= BB_PERIOD:
                sma = ohlcv["close"].rolling(BB_PERIOD).mean().iloc[-1]
                if not math.isnan(sma) and price >= sma:
                    reduce_position(ASSET, abs(size))
                    return
        return

    # Get OHLCV data
    ohlcv = get_futures_ohlcv(ASSET, "{interval}", max(BB_PERIOD, RSI_PERIOD) + 10)
    if ohlcv is None or len(ohlcv) < BB_PERIOD + 2:
        return

    closes = ohlcv["close"]

    # Bollinger Bands
    sma = closes.rolling(BB_PERIOD).mean()
    std = closes.rolling(BB_PERIOD).std()
    lower_band = sma - (BB_STD * std)

    if price >= lower_band.iloc[-1]:
        return  # Not below lower band

    # RSI confirmation
    rsi = compute_rsi(closes, RSI_PERIOD)
    current_rsi = rsi.iloc[-1]
    if math.isnan(current_rsi) or current_rsi > RSI_ENTRY:
        return  # Not oversold enough

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
    "risk_pct": 0.10,
    "bb_period": 20,
    "bb_std": 2.0,
    "rsi_period": 14,
    "rsi_entry": 30,
    "tp_pct": 0.025,
    "sl_pct": 0.015,
    "interval": "1h",
}


def generate(**kwargs) -> str:
    """Generate mean reversion strategy code with custom parameters."""
    params = {**DEFAULTS, **kwargs}
    return TEMPLATE.format(**params).strip()
