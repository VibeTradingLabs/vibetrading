"""
Grid Trading Strategy Template

Places limit buy/sell orders at fixed price intervals around the current price.
Profits from price oscillation within a range.

Parameters (customize via the constants at the top):
- ASSET: Trading pair (default: BTC)
- GRID_LEVELS: Number of grid levels above and below (default: 5)
- GRID_SPACING_PCT: Distance between grid levels as % of price (default: 0.005)
- ORDER_SIZE_PCT: Fraction of balance per grid order (default: 0.03)
"""

TEMPLATE = """
import math
from vibetrading import (
    vibe, get_perp_price, get_spot_price, get_perp_summary,
    buy, sell, my_spot_balance,
)

# ── Parameters ─────────────────────────────────────────────────────
ASSET = "{asset}"
GRID_LEVELS = {grid_levels}
GRID_SPACING_PCT = {grid_spacing_pct}
ORDER_SIZE_PCT = {order_size_pct}

# Track grid state
_grid_state = {{"initialized": False, "last_grid_center": 0.0}}

@vibe(interval="{interval}")
def grid_strategy():
    price = get_spot_price(ASSET)
    if math.isnan(price) or price <= 0:
        return

    balance = my_spot_balance("USDC")
    asset_balance = my_spot_balance(ASSET)

    # Re-center grid if price moved >3 grid spacings from center
    if _grid_state["initialized"]:
        drift = abs(price - _grid_state["last_grid_center"]) / price
        if drift < GRID_SPACING_PCT * 3:
            return  # Grid is still centered, no action needed

    _grid_state["initialized"] = True
    _grid_state["last_grid_center"] = price

    # Place buy orders below current price
    for i in range(1, GRID_LEVELS + 1):
        buy_price = price * (1 - GRID_SPACING_PCT * i)
        order_value = balance * ORDER_SIZE_PCT
        if order_value < 10:
            continue
        qty = order_value / buy_price
        buy(ASSET, qty, buy_price, order_type="limit")

    # Place sell orders above current price (if we hold the asset)
    if asset_balance > 0:
        sell_qty_per_level = asset_balance / GRID_LEVELS
        for i in range(1, GRID_LEVELS + 1):
            sell_price = price * (1 + GRID_SPACING_PCT * i)
            if sell_qty_per_level * sell_price < 10:
                continue
            sell(ASSET, sell_qty_per_level, sell_price, order_type="limit")
"""

DEFAULTS = {
    "asset": "BTC",
    "grid_levels": 5,
    "grid_spacing_pct": 0.005,
    "order_size_pct": 0.03,
    "interval": "1h",
}


def generate(**kwargs) -> str:
    """Generate grid strategy code with custom parameters."""
    params = {**DEFAULTS, **kwargs}
    return TEMPLATE.format(**params).strip()
