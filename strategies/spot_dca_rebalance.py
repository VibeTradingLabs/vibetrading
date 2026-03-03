"""
Spot DCA with Portfolio Rebalancing

Dollar-cost averages into a multi-asset portfolio, then periodically
rebalances to maintain target allocations. Simple, effective, low-stress.

Features:
- DCA on each candle (configurable amount)
- Target allocation rebalancing
- Only rebalances when drift exceeds threshold (reduces fees)
- Take-profit on the whole portfolio

Backtest: vibetrading backtest strategies/spot_dca_rebalance.py -i 1d
"""

import math

from vibetrading import (
    buy,
    get_spot_price,
    my_spot_balance,
    sell,
    vibe,
)

# ── Parameters ─────────────────────────────────────────────────────
PORTFOLIO = {
    "BTC": 0.50,  # 50% Bitcoin
    "ETH": 0.30,  # 30% Ethereum
    "SOL": 0.20,  # 20% Solana
}

DCA_AMOUNT = 50  # USDC per candle per asset (proportional to allocation)
TOTAL_DCA = 100  # Total USDC to DCA per period

# Rebalancing
REBALANCE_PERIOD = 168  # Every 168 candles (~1 week on 1h, ~168 days on 1d)
DRIFT_THRESHOLD = 0.05  # Only rebalance if allocation drifts >5%

# Take profit on total portfolio
TP_PCT = 0.20  # 20% total portfolio gain

# State
_candles = 0
_total_invested = 0.0


def get_portfolio_value():
    """Calculate total portfolio value in USDC."""
    total = my_spot_balance("USDC")
    for asset in PORTFOLIO:
        balance = my_spot_balance(asset)
        price = get_spot_price(asset)
        if not math.isnan(price) and balance > 0:
            total += balance * price
    return total


def get_allocations():
    """Calculate current allocation percentages."""
    total = get_portfolio_value()
    if total <= 0:
        return {}

    allocations = {}
    for asset in PORTFOLIO:
        balance = my_spot_balance(asset)
        price = get_spot_price(asset)
        if math.isnan(price):
            allocations[asset] = 0.0
        else:
            allocations[asset] = (balance * price) / total

    return allocations


@vibe(interval="1d")
def spot_dca_rebalance():
    global _candles, _total_invested

    _candles += 1

    usdc_balance = my_spot_balance("USDC")
    portfolio_value = get_portfolio_value()

    # ── Portfolio take profit ─────────────────────────────────────
    if _total_invested > 0 and portfolio_value > 0:
        total_return = (portfolio_value - _total_invested) / _total_invested
        if total_return >= TP_PCT:
            # Sell everything
            for asset in PORTFOLIO:
                balance = my_spot_balance(asset)
                price = get_spot_price(asset)
                if balance > 0 and not math.isnan(price) and balance * price >= 10:
                    sell(asset, balance, price, order_type="market")
            _total_invested = 0.0
            return

    # ── DCA buy ───────────────────────────────────────────────────
    if usdc_balance >= TOTAL_DCA:
        for asset, target_pct in PORTFOLIO.items():
            price = get_spot_price(asset)
            if math.isnan(price) or price <= 0:
                continue

            amount = TOTAL_DCA * target_pct
            qty = amount / price
            if amount >= 10:
                result = buy(asset, qty, price, order_type="market")
                if result.get("status") == "success":
                    _total_invested += amount

    # ── Periodic rebalancing ──────────────────────────────────────
    if _candles % REBALANCE_PERIOD != 0:
        return

    current_alloc = get_allocations()
    if not current_alloc:
        return

    # Check if any asset drifted beyond threshold
    max_drift = max(abs(current_alloc.get(asset, 0) - target) for asset, target in PORTFOLIO.items())

    if max_drift < DRIFT_THRESHOLD:
        return  # Within tolerance, skip rebalance

    # Rebalance: sell overweight, buy underweight
    total = get_portfolio_value()

    for asset, target_pct in PORTFOLIO.items():
        price = get_spot_price(asset)
        if math.isnan(price) or price <= 0:
            continue

        current_pct = current_alloc.get(asset, 0)
        diff_pct = current_pct - target_pct

        if abs(diff_pct) < DRIFT_THRESHOLD:
            continue

        diff_value = diff_pct * total
        qty = abs(diff_value) / price

        if abs(diff_value) < 10:
            continue

        if diff_pct > 0:
            # Overweight — sell
            balance = my_spot_balance(asset)
            qty = min(qty, balance)
            if qty * price >= 10:
                sell(asset, qty, price, order_type="market")
        else:
            # Underweight — buy
            if usdc_balance >= qty * price:
                buy(asset, qty, price, order_type="market")
                usdc_balance -= qty * price
