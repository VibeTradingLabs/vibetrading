"""
The @vibe decorator for registering trading strategy callbacks.

In the context of the BacktestEngine or LiveRunner, the @vibe decorator
registers a function as a strategy callback that gets executed at each
time step.

Usage::

    from vibetrading import vibe

    @vibe
    def my_strategy():
        price = get_price("BTC")
        if price < 50000:
            buy("BTC", 0.1, price)

    # Or with explicit interval:
    @vibe(interval="1h")
    def my_strategy():
        ...

Note: When used inside BacktestEngine or LiveRunner, the decorator is
replaced by a mock that registers callbacks. This standalone version
serves as a no-op marker for module-level usage.
"""

from typing import Callable, Optional


def vibe(interval_or_func=None, **kwargs):
    """
    Strategy decorator that marks a function as a trading strategy callback.

    Can be used as:
    - @vibe             (no parentheses)
    - @vibe()           (empty parentheses)
    - @vibe(interval="1h")  (with parameters)
    """
    if callable(interval_or_func):
        # Called as @vibe (no parentheses) - interval_or_func is the function
        func = interval_or_func
        func._vibe_interval = "1m"
        func._is_vibe_strategy = True
        return func
    else:
        # Called as @vibe() or @vibe(interval="1h")
        interval = interval_or_func if interval_or_func is not None else kwargs.get('interval', "1m")

        def actual_decorator(func: Callable):
            func._vibe_interval = interval
            func._is_vibe_strategy = True
            return func
        return actual_decorator
