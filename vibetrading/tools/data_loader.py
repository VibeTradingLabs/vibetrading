"""
Data loader for reading pre-downloaded market data from CSV cache.

Provides symbol mappings, cache filename generation, and CSV loading
utilities. All data is expected to be pre-downloaded (e.g. via
``vibetrading.tools.data_downloader``); if data is missing, a clear
error is raised.
"""

import os
import logging
from typing import Optional

import pandas as pd

from ..config import DATASET_DIR

logger = logging.getLogger(__name__)


# ── Default symbol mappings ───────────────────────────────────────────

DEFAULT_PERP_SYMBOLS = {
    "BTC": "BTC/USDT:USDT",
    "ETH": "ETH/USDT:USDT",
    "SOL": "SOL/USDT:USDT",
    "BNB": "BNB/USDT:USDT",
    "XRP": "XRP/USDT:USDT",
    "DOGE": "DOGE/USDT:USDT",
    "ADA": "ADA/USDT:USDT",
    "AVAX": "AVAX/USDT:USDT",
    "LINK": "LINK/USDT:USDT",
    "DOT": "DOT/USDT:USDT",
    "SUI": "SUI/USDT:USDT",
    "ARB": "ARB/USDT:USDT",
    "OP": "OP/USDT:USDT",
    "PEPE": "PEPE/USDT:USDT",
    "WIF": "WIF/USDT:USDT",
}

DEFAULT_SPOT_SYMBOLS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "BNB": "BNB/USDT",
    "XRP": "XRP/USDT",
    "DOGE": "DOGE/USDT",
    "ADA": "ADA/USDT",
    "AVAX": "AVAX/USDT",
    "LINK": "LINK/USDT",
    "DOT": "DOT/USDT",
    "SUI": "SUI/USDT",
    "ARB": "ARB/USDT",
    "OP": "OP/USDT",
    "PEPE": "PEPE/USDT",
    "WIF": "WIF/USDT",
}


# ── Cache filename generation ─────────────────────────────────────────

def generate_cache_filename(
    exchange: str,
    symbol: str,
    start_date: str,
    end_date: str,
    timeframe: str,
    data_dir: str = DATASET_DIR,
) -> str:
    """Generate a standardized cache filename for market data CSV."""
    sym_clean = symbol.replace("/", "_").replace(":", "_")
    start_clean = start_date.replace("-", "")
    end_clean = end_date.replace("-", "")
    fname = f"{exchange}_{sym_clean}_{start_clean}_{end_clean}_{timeframe}.csv"
    return os.path.join(data_dir, fname)


# ── CSV loading ───────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV cache file and return a properly indexed DataFrame.

    Args:
        path: Absolute path to the CSV file.

    Returns:
        DataFrame with a UTC ``timestamp`` index and OHLCV columns.
        Returns an empty DataFrame if the file is empty or invalid.
    """
    data = pd.read_csv(path)
    if len(data) == 0 or (len(data) == 1 and "error_marker" in data.columns):
        return pd.DataFrame()

    if "timestamp" in data.columns:
        if pd.api.types.is_numeric_dtype(data["timestamp"]):
            data["timestamp"] = pd.to_datetime(
                data["timestamp"], unit="ms", utc=True, errors="coerce"
            )
        else:
            data["timestamp"] = pd.to_datetime(
                data["timestamp"], utc=True, errors="coerce"
            )
        data = data.dropna(subset=["timestamp"])

    data.set_index("timestamp", inplace=True)
    data.sort_index(inplace=True)
    return data
