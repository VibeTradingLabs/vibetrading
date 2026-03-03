"""
Position sizing utilities for strategy development.

Provides common position sizing methods that strategies can use to
determine how much capital to allocate to each trade.

Usage::

    from vibetrading._utils.sizing import (
        kelly_size,
        fixed_fraction_size,
        volatility_adjusted_size,
        risk_per_trade_size,
    )

    # Kelly criterion
    qty = kelly_size(win_rate=0.55, avg_win=200, avg_loss=100, balance=10000, price=50000)

    # Fixed fraction of equity
    qty = fixed_fraction_size(fraction=0.02, balance=10000, price=50000, leverage=3)

    # Volatility-adjusted (inverse ATR)
    qty = volatility_adjusted_size(atr=500, balance=10000, price=50000, risk_pct=0.02)

    # Fixed risk per trade (risk $X on stop-loss distance)
    qty = risk_per_trade_size(balance=10000, risk_pct=0.01, entry=50000, stop_loss=49000)
"""


def kelly_size(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    balance: float,
    price: float,
    leverage: float = 1.0,
    max_fraction: float = 0.25,
    half_kelly: bool = True,
) -> float:
    """Calculate position size using the Kelly Criterion.

    The Kelly Criterion determines the optimal fraction of capital to risk
    on a bet with a known edge. Half-Kelly (default) is recommended for
    real trading to reduce variance.

    Args:
        win_rate: Historical win rate (0-1).
        avg_win: Average winning trade PnL (positive).
        avg_loss: Average losing trade PnL (positive, will be treated as loss).
        balance: Current account balance.
        price: Current asset price.
        leverage: Position leverage (default: 1).
        max_fraction: Maximum fraction of balance to risk (default: 0.25).
        half_kelly: Use half-Kelly for reduced variance (default: True).

    Returns:
        Position size in asset units. Returns 0 if no edge exists.
    """
    if avg_loss <= 0 or balance <= 0 or price <= 0:
        return 0.0

    loss_rate = 1 - win_rate
    win_loss_ratio = avg_win / avg_loss

    # Kelly formula: f* = (p * b - q) / b
    # where p = win_rate, q = loss_rate, b = win/loss ratio
    kelly_fraction = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio

    if kelly_fraction <= 0:
        return 0.0  # No edge

    if half_kelly:
        kelly_fraction *= 0.5

    # Cap at max_fraction
    kelly_fraction = min(kelly_fraction, max_fraction)

    position_value = balance * kelly_fraction * leverage
    return position_value / price


def fixed_fraction_size(
    fraction: float,
    balance: float,
    price: float,
    leverage: float = 1.0,
) -> float:
    """Calculate position size as a fixed fraction of account balance.

    The simplest sizing method — allocate a constant percentage of equity
    to each trade.

    Args:
        fraction: Fraction of balance to use (e.g. 0.02 = 2%).
        balance: Current account balance.
        price: Current asset price.
        leverage: Position leverage (default: 1).

    Returns:
        Position size in asset units.
    """
    if balance <= 0 or price <= 0 or fraction <= 0:
        return 0.0

    position_value = balance * fraction * leverage
    return position_value / price


def volatility_adjusted_size(
    atr: float,
    balance: float,
    price: float,
    risk_pct: float = 0.02,
    leverage: float = 1.0,
    atr_multiplier: float = 2.0,
) -> float:
    """Calculate position size adjusted for asset volatility using ATR.

    Sizes positions inversely proportional to volatility — larger positions
    in calm markets, smaller in volatile markets. This is the basis of
    turtle trading and many systematic strategies.

    Args:
        atr: Average True Range (absolute, same units as price).
        balance: Current account balance.
        price: Current asset price.
        risk_pct: Maximum risk per trade as fraction of balance (default: 0.02).
        leverage: Position leverage (default: 1).
        atr_multiplier: ATR units for stop distance (default: 2.0).

    Returns:
        Position size in asset units. Returns 0 if ATR is invalid.
    """
    if atr <= 0 or balance <= 0 or price <= 0:
        return 0.0

    risk_amount = balance * risk_pct
    stop_distance = atr * atr_multiplier

    # Position size = risk_amount / stop_distance (in asset units)
    qty = (risk_amount / stop_distance) * leverage

    return qty


def risk_per_trade_size(
    balance: float,
    risk_pct: float,
    entry: float,
    stop_loss: float,
    leverage: float = 1.0,
) -> float:
    """Calculate position size based on fixed risk per trade.

    Determines position size such that if the stop-loss is hit, the loss
    equals exactly ``risk_pct`` of the account balance.

    Args:
        balance: Current account balance.
        risk_pct: Risk per trade as fraction of balance (e.g. 0.01 = 1%).
        entry: Entry price.
        stop_loss: Stop-loss price.
        leverage: Position leverage (default: 1).

    Returns:
        Position size in asset units. Returns 0 if stop distance is invalid.
    """
    if balance <= 0 or entry <= 0 or risk_pct <= 0:
        return 0.0

    stop_distance = abs(entry - stop_loss)
    if stop_distance <= 0:
        return 0.0

    risk_amount = balance * risk_pct
    qty = (risk_amount / stop_distance) * leverage

    return qty


def max_position_size(
    balance: float,
    price: float,
    leverage: float = 1.0,
    max_exposure_pct: float = 0.95,
) -> float:
    """Calculate the maximum position size given balance and leverage.

    Useful as a sanity check to ensure position sizes from other methods
    don't exceed account capacity.

    Args:
        balance: Current account balance.
        price: Current asset price.
        leverage: Position leverage (default: 1).
        max_exposure_pct: Maximum fraction of leveraged balance to use (default: 0.95).

    Returns:
        Maximum position size in asset units.
    """
    if balance <= 0 or price <= 0:
        return 0.0

    return (balance * leverage * max_exposure_pct) / price
