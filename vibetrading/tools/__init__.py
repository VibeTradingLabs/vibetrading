"""
Tools for data acquisition and external integrations.

These are standalone utilities meant to be run before backtesting or
live trading.  They are intentionally decoupled from the core library
so the core can assume data is already available on disk.
"""

from .data_downloader import download_data
from .data_loader import (
    DEFAULT_PERP_SYMBOLS,
    DEFAULT_SPOT_SYMBOLS,
    generate_cache_filename,
    load_csv,
)

__all__ = [
    "download_data",
    "DEFAULT_PERP_SYMBOLS",
    "DEFAULT_SPOT_SYMBOLS",
    "generate_cache_filename",
    "load_csv",
]
