"""
Configuration settings for the VibeTrading framework.

Provides centralized path management and a multi-exchange credential
registry.  Values are read from environment variables (with fallbacks)
and can be overridden via ``.env`` / ``.env.local`` files.

Exchange configuration uses CCXT-compatible field names so the dict
can be passed directly to ``ccxt.<exchange>(EXCHANGES["binance"])``::

    from vibetrading.config import EXCHANGES

    EXCHANGES["binance"] = {
        "apiKey": "...",
        "secret": "...",
    }
    EXCHANGES["okx"] = {
        "apiKey": "...",
        "secret": "...",
        "password": "...",
    }
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_env_local_path = Path('.env.local')
if _env_local_path.exists():
    load_dotenv(dotenv_path=_env_local_path, override=True)


def str_to_bool(value) -> bool:
    """Convert string value to boolean."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() in ('true', '1', 'yes', 'on')


# ── Package directory paths ───────────────────────────────────────────
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.getcwd(), "vibetrading", "dataset")

# ── Global settings ───────────────────────────────────────────────────
DEFAULT_EXCHANGE = os.environ.get('VIBETRADING_DEFAULT_EXCHANGE', 'binance')

# ── Exchange registry (CCXT-compatible) ───────────────────────────────
#
# Keys are exchange identifiers, values are dicts that can be passed
# directly to ccxt.<exchange>(config).
#
# {
#     "binance": {"apiKey": "...", "secret": "..."},
#     "okx":     {"apiKey": "...", "secret": "...", "password": "..."},
# }
EXCHANGES: dict[str, dict] = {
    "binance": {
        "apiKey": os.environ.get('BINANCE_API_KEY', ''),
        "secret": os.environ.get('BINANCE_API_SECRET', ''),
        "options": {
            "defaultType": "swap",  # 永续合约类型
        }
    },
    "hyperliquid": {
        "walletAddress": os.environ.get('HYPERLIQUID_WALLET_ADDRESS', ''),
        "privateKey": os.environ.get('HYPERLIQUID_PRIVATE_KEY', ''),
    }
}


# ── Directory initialization ──────────────────────────────────────────

def ensure_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(DATASET_DIR, exist_ok=True)


ensure_directories()
