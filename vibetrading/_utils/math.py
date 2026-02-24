"""
Mathematical utility functions for trading operations.
"""

import math
from typing import Union


def truncate_quantity(quantity: float, decimal_places: int) -> float:
    """
    Safely truncate quantity to specified decimal places using floor operation.
    This prevents overselling by always rounding down, never up.

    Args:
        quantity: The quantity to truncate
        decimal_places: Number of decimal places to keep

    Returns:
        Truncated quantity (always rounded down)

    Examples:
        >>> truncate_quantity(1.23956789, 2)
        1.23
        >>> truncate_quantity(0.999999, 3)
        0.999
        >>> truncate_quantity(10.5555, 1)
        10.5
    """
    if quantity < 0:
        raise ValueError("Quantity must be non-negative")
    if decimal_places < 0:
        raise ValueError("Decimal places must be non-negative")

    multiplier = 10 ** decimal_places
    return math.floor(quantity * multiplier) / multiplier


def format_hyperliquid_price(price: float, is_spot: bool = True) -> float:
    """
    Format price for Hyperliquid API compliance.

    Hyperliquid requires specific price formatting:
    - Remove trailing zeros (e.g., 12345.0 -> 12345, 0.123450 -> 0.12345)
    - Spot markets: max 8 decimal places, 5 significant figures
    - Perp markets: max 6 decimal places, 5 significant figures

    Args:
        price: The price to format
        is_spot: True for spot markets, False for perpetual markets

    Returns:
        Properly formatted price for Hyperliquid API
    """
    if price <= 0:
        raise ValueError("Price must be positive")

    max_decimals = 8 if is_spot else 6

    if price >= 1:
        integer_digits = len(str(int(price)))
        decimal_places = max(0, 5 - integer_digits)
        formatted_price = round(price, min(decimal_places, max_decimals))
    else:
        sig_figs = 5
        formatted_price = round(price, sig_figs - int(math.floor(math.log10(abs(price)))) - 1)
        formatted_price = round(formatted_price, max_decimals)

    price_str = f"{formatted_price:.{max_decimals}f}".rstrip('0').rstrip('.')
    return float(price_str)
