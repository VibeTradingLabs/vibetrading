"""
Exchange sandbox abstractions and factory.

Usage::

    import vibetrading.sandbox

    # Create a live exchange sandbox
    sb = vibetrading.sandbox.create("hyperliquid", api_key="...", api_secret="...")
    price = sb.get_price("BTC")

    # Run a strategy live
    runner = vibetrading.sandbox.LiveRunner(sb, interval="1m")
    runner.load_strategy(code)
    await runner.start()

    # Type annotations
    from vibetrading.sandbox import SandboxBase
    def my_func(sb: SandboxBase): ...
"""

from ._core.sandbox_base import (
    VibeSandboxBase as SandboxBase,
    SUPPORTED_INTERVALS,
    SUPPORTED_LEVERAGE,
)
from ._core.live_runner import LiveRunner
from ._exchanges import create_sandbox as create, SUPPORTED_EXCHANGES

__all__ = [
    "SandboxBase",
    "SUPPORTED_INTERVALS",
    "SUPPORTED_LEVERAGE",
    "LiveRunner",
    "create",
    "SUPPORTED_EXCHANGES",
]
