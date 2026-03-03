"""
Live trading — run strategies against real exchanges.

Usage::

    import vibetrading.live

    # Start live trading on Hyperliquid
    await vibetrading.live.start(
        strategy_code,
        exchange="hyperliquid",
        api_key="0x...",         # wallet address
        api_secret="0x...",      # private key
        interval="1m",
    )

    # Or with more control
    import vibetrading.sandbox

    sandbox = vibetrading.sandbox.create("hyperliquid", api_key="0x...", api_secret="0x...")
    runner = vibetrading.sandbox.LiveRunner(sandbox, interval="1m")
    runner.load_strategy(code)
    await runner.start()
"""

import asyncio
from typing import Any

from ._core.live_runner import LiveRunner
from ._exchanges import create_sandbox


async def start(
    strategy_code: str,
    *,
    exchange: str = "hyperliquid",
    api_key: str | None = None,
    api_secret: str | None = None,
    interval: str = "1m",
    mode: str = "live",
    **kwargs: Any,
) -> None:
    """
    Start live trading with a strategy.

    This is a blocking coroutine that runs the strategy in a loop.
    Use Ctrl+C or runner.stop() to stop.

    Args:
        strategy_code: Python code with a @vibe-decorated strategy function.
        exchange: Exchange name (hyperliquid, paradex, lighter, aster).
        api_key: API key or wallet address.
        api_secret: API secret or private key.
        interval: Execution interval (e.g. "1m", "5m", "1h").
        mode: 'live' for real trading, 'paper' for paper trading.
        **kwargs: Additional exchange-specific parameters.
    """
    sandbox = create_sandbox(
        exchange,
        api_key=api_key,
        api_secret=api_secret,
        mode=mode,
        **kwargs,
    )
    runner = LiveRunner(sandbox, interval=interval)
    runner.load_strategy(strategy_code)
    await runner.start()


def start_sync(
    strategy_code: str,
    *,
    exchange: str = "hyperliquid",
    api_key: str | None = None,
    api_secret: str | None = None,
    interval: str = "1m",
    mode: str = "live",
    **kwargs: Any,
) -> None:
    """
    Start live trading (synchronous wrapper).

    Convenience function that wraps start() for non-async contexts.
    Use Ctrl+C to stop.

    Args:
        Same as start().
    """
    asyncio.run(
        start(
            strategy_code,
            exchange=exchange,
            api_key=api_key,
            api_secret=api_secret,
            interval=interval,
            mode=mode,
            **kwargs,
        )
    )


__all__ = [
    "start",
    "start_sync",
]
