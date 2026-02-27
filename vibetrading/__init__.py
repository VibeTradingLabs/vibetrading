"""
VibeTrading - Agent-first trading framework for prompt-to-strategy generation.

Describe strategies in natural language. Generate executable Python code.
Backtest and deploy to any supported exchange with the same code.

Public modules::

    import vibetrading.strategy    # generate, validate & analyze strategies
    import vibetrading.backtest    # backtest engine
    import vibetrading.evolution   # iterative strategy evolution
    import vibetrading.sandbox     # exchange sandboxes & live runner
    import vibetrading.tools       # data download & loading
    import vibetrading.models      # order & position data models

Quick start::

    import vibetrading

    # One-shot: generate and backtest
    code = vibetrading.strategy.generate("BTC momentum with RSI oversold entry")
    results = vibetrading.backtest.run(code)

    # Full loop: evolve iteratively via LLM feedback
    result = vibetrading.evolve("BTC momentum with RSI", iterations=3, model="gpt-4o")
    print(result.best_code)

The ``vibe`` decorator is available at the package root for strategy code::

    from vibetrading import vibe

    @vibe
    def my_strategy():
        ...
"""

__version__ = "0.1.6"

from ._core.decorator import vibe
from .evolution import evolve

__all__ = [
    "__version__",
    "vibe",
    "evolve",
]
