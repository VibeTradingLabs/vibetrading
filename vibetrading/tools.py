"""
Data acquisition and loading utilities.

Usage::

    import vibetrading.tools

    # Download historical data from an exchange
    data = vibetrading.tools.download_data(
        ["BTC", "ETH"],
        exchange="binance",
        interval="1h",
        start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2025, 6, 1, tzinfo=timezone.utc),
    )

    # Load cached CSV data
    df = vibetrading.tools.load_csv(path)
"""

from ._tools.data_downloader import download_data
from ._tools.data_loader import (
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
