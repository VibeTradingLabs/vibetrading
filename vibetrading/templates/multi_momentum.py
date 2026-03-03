"""
Multi-Asset Momentum Strategy Template

Trades multiple assets simultaneously using momentum signals.
Allocates capital equally across assets with active signals.
Each asset has independent TP/SL management.

Parameters (customize via the constants at the top):
- ASSETS: List of assets to trade
- LEVERAGE: Position leverage per asset
- RISK_PCT_PER_ASSET: Fraction of margin allocated per asset
- SMA_FAST / SMA_SLOW: Moving average periods
- TP_PCT / SL_PCT: Take-profit and stop-loss per position
"""

TEMPLATE = '''
import math
from vibetrading import (
    vibe, get_perp_price, get_futures_ohlcv, get_perp_summary,
    get_perp_position, set_leverage, long, reduce_position,
)

# ── Parameters ─────────────────────────────────────────────────────
ASSETS = {assets}
LEVERAGE = {leverage}
RISK_PCT_PER_ASSET = {risk_pct_per_asset}
SMA_FAST = {sma_fast}
SMA_SLOW = {sma_slow}
TP_PCT = {tp_pct}
SL_PCT = {sl_pct}

def compute_rsi(closes, period=14):
    """Compute RSI from a pandas Series."""
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    if (loss == 0).all():
        return gain * 0 + 50  # neutral
    rs = gain / loss.replace(0, float("inf"))
    return 100 - (100 / (1 + rs))

@vibe(interval="{interval}")
def multi_momentum_strategy():
    summary = get_perp_summary()
    total_margin = summary.get("available_margin", 0)

    for asset in ASSETS:
        price = get_perp_price(asset)
        if math.isnan(price) or price <= 0:
            continue

        # Manage existing position
        position = get_perp_position(asset)
        if position and position.get("size", 0) != 0:
            entry = position["entry_price"]
            size = position["size"]
            pnl_pct = (price - entry) / entry if entry > 0 else 0

            if size > 0:
                if pnl_pct >= TP_PCT or pnl_pct <= -SL_PCT:
                    reduce_position(asset, abs(size))
            continue

        # Check for entry signal
        ohlcv = get_futures_ohlcv(asset, "{interval}", SMA_SLOW + 5)
        if ohlcv is None or len(ohlcv) < SMA_SLOW + 2:
            continue

        closes = ohlcv["close"]
        sma_fast = closes.rolling(SMA_FAST).mean()
        sma_slow = closes.rolling(SMA_SLOW).mean()

        # Require fresh bullish crossover
        if sma_fast.iloc[-1] <= sma_slow.iloc[-1]:
            continue
        if sma_fast.iloc[-2] >= sma_slow.iloc[-2]:
            continue

        # RSI filter
        rsi = compute_rsi(closes)
        if math.isnan(rsi.iloc[-1]) or rsi.iloc[-1] > 75:
            continue

        # Position sizing: equal allocation across assets
        margin_per_asset = total_margin * RISK_PCT_PER_ASSET
        if margin_per_asset < 50:
            continue

        set_leverage(asset, LEVERAGE)
        qty = (margin_per_asset * LEVERAGE) / price
        if qty * price >= 15:
            long(asset, qty, price, order_type="market")
'''

DEFAULTS = {
    "assets": '["BTC", "ETH", "SOL"]',
    "leverage": 3,
    "risk_pct_per_asset": 0.10,
    "sma_fast": 10,
    "sma_slow": 30,
    "tp_pct": 0.04,
    "sl_pct": 0.02,
    "interval": "1h",
}


def generate(**kwargs) -> str:
    """Generate multi-asset momentum strategy code with custom parameters."""
    params = {**DEFAULTS, **kwargs}
    # Handle assets as list or string
    if isinstance(params["assets"], list):
        params["assets"] = repr(params["assets"])
    return TEMPLATE.format(**params).strip()
