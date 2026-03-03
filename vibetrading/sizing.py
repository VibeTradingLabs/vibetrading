"""
Position sizing utilities for strategy development.

Usage::

    from vibetrading.sizing import (
        kelly_size,
        fixed_fraction_size,
        volatility_adjusted_size,
        risk_per_trade_size,
        max_position_size,
    )

    # Kelly criterion (half-Kelly by default)
    qty = kelly_size(win_rate=0.55, avg_win=200, avg_loss=100, balance=10000, price=50000)

    # Fixed fraction of equity
    qty = fixed_fraction_size(fraction=0.02, balance=10000, price=50000, leverage=3)

    # Volatility-adjusted using ATR
    qty = volatility_adjusted_size(atr=500, balance=10000, price=50000, risk_pct=0.02)

    # Fixed risk per trade (risk $X on stop-loss distance)
    qty = risk_per_trade_size(balance=10000, risk_pct=0.01, entry=50000, stop_loss=49000)

    # Sanity check for max position
    max_qty = max_position_size(balance=10000, price=50000, leverage=3)
"""

from ._utils.sizing import (
    fixed_fraction_size,
    kelly_size,
    max_position_size,
    risk_per_trade_size,
    volatility_adjusted_size,
)

__all__ = [
    "fixed_fraction_size",
    "kelly_size",
    "max_position_size",
    "risk_per_trade_size",
    "volatility_adjusted_size",
]
