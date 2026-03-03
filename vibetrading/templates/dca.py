"""
Dollar-Cost Averaging (DCA) Strategy Template

Buys a fixed dollar amount of an asset at each interval.
Optionally takes profit when cumulative return exceeds a threshold.

Parameters (customize via the constants at the top):
- ASSET: Trading pair (default: BTC)
- BUY_AMOUNT: USDC to spend per interval (default: 50)
- TP_PCT: Take-profit threshold for selling entire position (default: 0.10)
"""

TEMPLATE = """
import math
from vibetrading import (
    vibe, get_spot_price, my_spot_balance, buy, sell,
)

# ── Parameters ─────────────────────────────────────────────────────
ASSET = "{asset}"
BUY_AMOUNT = {buy_amount}
TP_PCT = {tp_pct}

# Track cost basis
_dca_state = {{"total_cost": 0.0, "total_qty": 0.0}}

@vibe(interval="{interval}")
def dca_strategy():
    price = get_spot_price(ASSET)
    if math.isnan(price) or price <= 0:
        return

    balance = my_spot_balance("USDC")
    asset_balance = my_spot_balance(ASSET)

    # Check for take-profit on existing position
    if _dca_state["total_qty"] > 0 and asset_balance > 0:
        avg_cost = _dca_state["total_cost"] / _dca_state["total_qty"]
        if avg_cost > 0:
            pnl_pct = (price - avg_cost) / avg_cost
            if pnl_pct >= TP_PCT:
                sell(ASSET, asset_balance, price, order_type="market")
                _dca_state["total_cost"] = 0.0
                _dca_state["total_qty"] = 0.0
                return

    # DCA buy
    if balance >= BUY_AMOUNT:
        qty = BUY_AMOUNT / price
        if qty * price >= 10:
            buy(ASSET, qty, price, order_type="market")
            _dca_state["total_cost"] += BUY_AMOUNT
            _dca_state["total_qty"] += qty
"""

DEFAULTS = {
    "asset": "BTC",
    "buy_amount": 50,
    "tp_pct": 0.10,
    "interval": "1d",
}


def generate(**kwargs) -> str:
    """Generate DCA strategy code with custom parameters."""
    params = {**DEFAULTS, **kwargs}
    return TEMPLATE.format(**params).strip()
