"""
VibeTrading - Agent-first trading framework for prompt-to-strategy generation.

Describe strategies in natural language. Generate executable Python code.
Backtest and deploy to any supported exchange with the same code.

Public modules::

    import vibetrading.strategy   # generate & validate strategies
    import vibetrading.backtest   # backtest engine
    import vibetrading.sandbox    # exchange sandboxes & live runner
    import vibetrading.tools      # data download & loading
    import vibetrading.models     # order & position data models

Quick start::

    import vibetrading.strategy
    import vibetrading.backtest

    code = vibetrading.strategy.generate("BTC momentum with RSI oversold entry")
    results = vibetrading.backtest.run(code)

The ``vibe`` decorator is available at the package root for strategy code::

    from vibetrading import vibe

    @vibe
    def my_strategy():
        ...
"""

__version__ = "0.1.0"

from ._core.decorator import vibe

__all__ = [
    "__version__",
    "vibe",
]
