"""
Exchange sandbox implementations.

The factory function `create_sandbox()` is the recommended way to instantiate
an exchange sandbox. Exchange SDKs are lazily imported so they're only
required when actually creating a sandbox for that exchange.
"""

from typing import Optional

from .._core.sandbox_base import VibeSandboxBase


SUPPORTED_EXCHANGES = [
    "hyperliquid",
    "extended",
    "paradex",
    "lighter",
    "aster",
]


def create_sandbox(
    exchange: str,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    mode: str = "live",
    **kwargs,
) -> VibeSandboxBase:
    """
    Factory function to create an exchange sandbox instance.

    Args:
        exchange: Exchange identifier (e.g. 'hyperliquid', 'extended', 'paradex', 'lighter', 'aster')
        api_key: API key / wallet address
        api_secret: API secret / private key
        mode: 'live' or 'paper'
        **kwargs: Exchange-specific parameters

    Returns:
        A VibeSandboxBase instance for the specified exchange

    Raises:
        ValueError: If exchange is not supported
        ImportError: If required exchange SDK is not installed
    """
    exchange = exchange.lower()

    if exchange == "hyperliquid":
        from .hyperliquid import HyperliquidSandbox
        return HyperliquidSandbox(
            api_key=api_key, api_secret=api_secret, mode=mode, **kwargs
        )
    elif exchange == "extended":
        from .extended import ExtendedSandbox
        return ExtendedSandbox(
            api_key=api_key, api_secret=api_secret, mode=mode, **kwargs
        )
    elif exchange == "paradex":
        from .paradex import ParadexSandbox
        return ParadexSandbox(
            api_key=api_key, api_secret=api_secret, mode=mode, **kwargs
        )
    elif exchange == "lighter":
        from .lighter import LighterSandbox
        return LighterSandbox(
            api_key=api_key, api_secret=api_secret, mode=mode, **kwargs
        )
    elif exchange in ("aster", "aster_testnet"):
        from .aster import AsterSandbox
        testnet = exchange == "aster_testnet"
        return AsterSandbox(
            api_key=api_key, api_secret=api_secret, mode=mode,
            testnet=testnet, **kwargs
        )
    else:
        raise ValueError(
            f"Unsupported exchange: {exchange}. "
            f"Supported: {SUPPORTED_EXCHANGES}"
        )
