"""
A data downloader for fetching historical cryptocurrency data.

Downloads OHLCV, funding rate, and open interest data from exchanges
via the CCXT library and saves them as CSV files to the dataset directory.

Usage::

    from vibetrading.tools.data_downloader import download_data

    data = download_data(
        ["BTC", "ETH"],
        exchange="binance",
        start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2025, 6, 1, tzinfo=timezone.utc),
        interval="1h",
    )
"""

import os
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd

from .._config import DATASET_DIR, DEFAULT_EXCHANGE, EXCHANGES
from .data_loader import (
    DEFAULT_PERP_SYMBOLS,
    DEFAULT_SPOT_SYMBOLS,
    generate_cache_filename,
    load_csv,
)

logger = logging.getLogger(__name__)


def download_data(
    assets: List[str],
    *,
    exchange: str = "binance",
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    interval: str = "1h",
    market_type: str = "perp",
    data_dir: str = DATASET_DIR,
    force_refresh: bool = False,
    include_funding: bool = True,
    include_oi: bool = True,
    proxy: Optional[str] = None,
    timeout: int = 30000,
) -> Dict[str, pd.DataFrame]:
    """
    Download historical market data and save to CSV cache.

    This must be called before running a backtest if data has not been
    downloaded yet.  Downloaded data is saved as CSV files in *data_dir*
    and also returned as DataFrames.

    Args:
        assets: List of asset symbols (e.g., ``["BTC", "ETH", "SOL"]``).
        exchange: CCXT exchange identifier (default: ``"binance"``).
        start_time: Start time for data (default: ``end_time - 180 days``).
        end_time: End time for data (default: current UTC time).
        interval: Candle interval (e.g., ``"1h"``, ``"5m"``, ``"1d"``).
        market_type: ``"perp"`` for futures or ``"spot"`` for spot data.
        data_dir: Directory to save downloaded CSV files.
        force_refresh: Re-download even if cached data exists.
        include_funding: Fetch and merge funding rate data (perp only).
        include_oi: Fetch and merge open interest data (perp only).
        proxy: HTTPS proxy URL (e.g., ``"http://127.0.0.1:7890"``).
            Falls back to the ``HTTPS_PROXY`` / ``HTTP_PROXY`` env var.
        timeout: Request timeout in milliseconds (default: 30 000).

    Returns:
        Dict mapping ``"ASSET/INTERVAL"`` keys to DataFrames with columns
        ``[open, high, low, close, volume, fundingRate, openInterest]``.
        (funding/OI columns are NaN for spot data).
        DataFrame index is ``timestamp`` (UTC).
    """
    import numpy as np

    if end_time is None:
        end_time = datetime.now(tz=timezone.utc)
    if start_time is None:
        start_time = end_time - timedelta(days=30)

    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)

    start_date = start_time.strftime("%Y-%m-%d")
    end_date = end_time.strftime("%Y-%m-%d")

    # Extend range for lookback data
    fetch_start = start_time - timedelta(days=3)
    fetch_end = end_time + timedelta(days=1)
    start_ms = int(fetch_start.timestamp() * 1000)
    end_ms = int(fetch_end.timestamp() * 1000)

    os.makedirs(data_dir, exist_ok=True)
    tool = _CcxtClient(exchange_id=exchange, proxy=proxy, timeout=timeout)

    symbol_map = DEFAULT_PERP_SYMBOLS if market_type == "perp" else DEFAULT_SPOT_SYMBOLS
    is_futures = market_type == "perp"

    results: Dict[str, pd.DataFrame] = {}

    for asset in assets:
        asset = asset.upper()
        symbol = symbol_map.get(asset)
        if symbol is None:
            symbol = f"{asset}/USDT:USDT" if is_futures else f"{asset}/USDT"

        cache_file = generate_cache_filename(
            exchange, symbol, start_date, end_date, interval, data_dir,
        )
        key = f"{asset}/{interval}"

        # Check cache
        if not force_refresh and os.path.exists(cache_file):
            try:
                data = load_csv(cache_file)
                if not data.empty:
                    results[key] = data
                    print(f"  [cached] {asset} {interval}: {len(data)} rows")
                    continue
            except Exception:
                pass

        # Download fresh data
        print(f"  Downloading {asset} ({symbol}) {interval} from {exchange}...")
        try:
            ohlcv_df = tool.fetch_ohlcv(
                symbol=symbol,
                timeframe=interval,
                start_ms=start_ms,
                end_ms=end_ms,
            )
            if ohlcv_df.empty:
                print(f"  WARNING: No data for {asset} ({symbol})")
                results[key] = pd.DataFrame()
                continue

            ohlcv_df["timestamp"] = pd.to_datetime(
                ohlcv_df["timestamp"], unit="ms", utc=True,
            )
            ohlcv_df.set_index("timestamp", inplace=True)

            start_filter = pd.to_datetime(start_date, utc=True) - timedelta(days=3)
            end_filter = pd.to_datetime(end_date, utc=True) + timedelta(days=1)
            data = ohlcv_df[
                (ohlcv_df.index >= start_filter) & (ohlcv_df.index < end_filter)
            ].copy()

            if data.empty:
                print(f"  WARNING: No data in date range for {asset}")
                results[key] = pd.DataFrame()
                continue

            # Merge funding rate for futures
            if is_futures and include_funding:
                data = _merge_funding_rate(
                    tool, symbol, data, start_filter, end_filter, fetch_start,
                )

            # Merge open interest for futures
            if is_futures and include_oi:
                data = _merge_open_interest(
                    tool, symbol, data, start_filter, end_filter, fetch_start,
                    interval,
                )

            # Ensure columns exist
            if "fundingRate" not in data.columns:
                data["fundingRate"] = 0.0
            if "openInterest" not in data.columns:
                data["openInterest"] = np.nan

            # Save to cache
            data.reset_index(inplace=True)
            data.to_csv(cache_file, index=False)
            data.set_index("timestamp", inplace=True)
            data.sort_index(inplace=True)

            results[key] = data
            print(f"  [downloaded] {asset} {interval}: {len(data)} rows")

        except Exception as e:
            print(f"  ERROR downloading {asset}: {e}")
            results[key] = pd.DataFrame()

    return results


# ── Internal helpers ──────────────────────────────────────────────────


def _merge_funding_rate(
    tool: "_CcxtClient",
    symbol: str,
    data: pd.DataFrame,
    start_filter: pd.Timestamp,
    end_filter: pd.Timestamp,
    fetch_start: datetime,
) -> pd.DataFrame:
    """Merge funding rate data into OHLCV DataFrame."""
    try:
        days_back = (datetime.now(tz=timezone.utc) - fetch_start).days + 4
        fr_df = tool.fetch_funding_rate_history(symbol=symbol, days_back=days_back)
        if not fr_df.empty:
            fr_df["timestamp"] = pd.to_datetime(
                fr_df["timestamp"], unit="ms", utc=True,
            )
            fr_df.set_index("timestamp", inplace=True)
            fr_df = fr_df[
                (fr_df.index >= start_filter) & (fr_df.index < end_filter)
            ]
            if not fr_df.empty:
                ohlcv_m = data.reset_index()
                fr_m = fr_df.reset_index()
                merged = pd.merge_asof(
                    ohlcv_m.sort_values("timestamp"),
                    fr_m[["timestamp", "fundingRate"]].sort_values("timestamp"),
                    on="timestamp",
                    direction="backward",
                )
                merged.set_index("timestamp", inplace=True)
                data = merged
            data["fundingRate"] = data.get(
                "fundingRate", pd.Series(dtype=float)
            ).fillna(0.0)
        else:
            data["fundingRate"] = 0.0
    except Exception:
        data["fundingRate"] = 0.0
    return data


def _merge_open_interest(
    tool: "_CcxtClient",
    symbol: str,
    data: pd.DataFrame,
    start_filter: pd.Timestamp,
    end_filter: pd.Timestamp,
    fetch_start: datetime,
    interval: str = "1h",
) -> pd.DataFrame:
    """Merge open interest data into OHLCV DataFrame."""
    import numpy as np

    try:
        days_back = (datetime.now(tz=timezone.utc) - fetch_start).days + 4
        oi_df = tool.fetch_open_interest_history(
            symbol=symbol, days_back=days_back, timeframe=interval,
        )
        if not oi_df.empty:
            oi_df["timestamp"] = pd.to_datetime(
                oi_df["timestamp"], unit="ms", utc=True,
            )
            oi_df.set_index("timestamp", inplace=True)
            oi_filt = oi_df[
                (oi_df.index >= start_filter) & (oi_df.index < end_filter)
            ]
            if not oi_filt.empty:
                ohlcv_m = data.reset_index()
                oi_m = oi_filt.reset_index()
                merged = pd.merge_asof(
                    ohlcv_m.sort_values("timestamp"),
                    oi_m[["timestamp", "openInterest"]].sort_values("timestamp"),
                    on="timestamp",
                    direction="backward",
                )
                merged.set_index("timestamp", inplace=True)
                data = merged
                data["openInterest"] = data["openInterest"].ffill()
            else:
                data["openInterest"] = np.nan
        else:
            data["openInterest"] = np.nan
    except Exception:
        data["openInterest"] = np.nan
    return data


# ── Low-level CCXT client ─────────────────────────────────────────────


class _CcxtClient:
    """Low-level CCXT wrapper for historical data fetching."""

    def __init__(
        self,
        exchange_id: str | None = None,
        proxy: str | None = None,
        timeout: int = 30000,
    ):
        self.exchange_id = (exchange_id or DEFAULT_EXCHANGE).lower()
        self._proxy = proxy
        self._timeout = timeout
        self._exchange = None

    _AUTH_KEYS = {"apiKey", "secret", "password", "walletAddress", "privateKey"}

    def _resolve_proxy(self) -> str | None:
        if self._proxy:
            return self._proxy
        return (
            os.environ.get("HTTPS_PROXY")
            or os.environ.get("https_proxy")
            or os.environ.get("HTTP_PROXY")
            or os.environ.get("http_proxy")
        )

    def _get_exchange(self):
        """Lazily initialize the CCXT exchange client (public endpoints only)."""
        if self._exchange is None:
            import ccxt

            config: dict = {
                "enableRateLimit": True,
                "timeout": self._timeout,
            }
            proxy = self._resolve_proxy()
            if proxy:
                config["httpsProxy"] = proxy
                logger.info("Using proxy: %s", proxy)

            exchange_cfg = EXCHANGES.get(self.exchange_id, {})
            config.update(
                {k: v for k, v in exchange_cfg.items() if k not in self._AUTH_KEYS}
            )

            exchange_class = getattr(ccxt, self.exchange_id, None)
            if exchange_class is None:
                exchange_class = ccxt.binance
            self._exchange = exchange_class(config)

            try:
                self._exchange.load_markets()
            except ccxt.RequestTimeout:
                raise ConnectionError(
                    f"Cannot reach {self.exchange_id} API (request timed out). "
                    f"If you are behind a firewall / in a restricted region, "
                    f"set the `proxy` parameter or the HTTPS_PROXY env var.\n"
                    f"  Example: download_data([...], proxy='http://127.0.0.1:7890')\n"
                    f"  Or:      export HTTPS_PROXY=http://127.0.0.1:7890"
                ) from None
            except ccxt.NetworkError as exc:
                raise ConnectionError(
                    f"Network error connecting to {self.exchange_id}: {exc}\n"
                    f"Check your internet connection or configure a proxy via "
                    f"the `proxy` parameter or the HTTPS_PROXY env var."
                ) from None
        return self._exchange

    @staticmethod
    def _interval_ms(timeframe: str) -> int:
        mapping = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "4h": 240, "1d": 1440, "3d": 4320,
        }
        if timeframe not in mapping:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        return mapping[timeframe] * 60 * 1000

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_ms: int = 0,
        end_ms: int = 0,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a date range."""
        exchange = self._get_exchange()
        if not start_ms or not end_ms:
            raise ValueError("start_ms and end_ms are required")

        all_data = []
        since = start_ms
        limit = 1000
        interval_ms = self._interval_ms(timeframe)

        while since < end_ms:
            try:
                ohlcv = exchange.fetch_ohlcv(
                    symbol, timeframe, since=since, limit=limit,
                )
                if not ohlcv:
                    break
                all_data.extend(ohlcv)
                since = ohlcv[-1][0] + interval_ms
                if len(ohlcv) < limit:
                    break
                time.sleep(exchange.rateLimit / 1000)
            except Exception as e:
                logger.warning("Error fetching OHLCV for %s: %s", symbol, e)
                break

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(
            all_data,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df = df[(df["timestamp"] >= start_ms) & (df["timestamp"] <= end_ms)]
        df = (
            df.drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        return df

    def fetch_funding_rate_history(
        self, symbol: str, days_back: int = 30,
    ) -> pd.DataFrame:
        """Fetch historical funding rate data."""
        exchange = self._get_exchange()
        since_ms = int(
            (datetime.now(tz=timezone.utc) - timedelta(days=days_back)).timestamp()
            * 1000
        )
        try:
            if hasattr(exchange, "fetch_funding_rate_history"):
                all_rates = []
                since = since_ms
                while True:
                    rates = exchange.fetch_funding_rate_history(
                        symbol, since=since, limit=1000,
                    )
                    if not rates:
                        break
                    all_rates.extend(rates)
                    since = rates[-1]["timestamp"] + 1
                    if len(rates) < 1000:
                        break
                    time.sleep(exchange.rateLimit / 1000)

                if all_rates:
                    return pd.DataFrame(
                        [
                            {
                                "timestamp": r["timestamp"],
                                "fundingRate": r.get("fundingRate", 0.0),
                            }
                            for r in all_rates
                        ]
                    )
        except Exception as e:
            logger.warning("Error fetching funding rates for %s: %s", symbol, e)
        return pd.DataFrame()

    _OI_MAX_DAYS: dict[str, int] = {
        "5m": 2, "15m": 30, "30m": 30, "1h": 30,
        "2h": 90, "4h": 90, "6h": 90, "12h": 90, "1d": 180,
    }

    def fetch_open_interest_history(
        self,
        symbol: str,
        days_back: int = 30,
        timeframe: str = "1h",
    ) -> pd.DataFrame:
        """Fetch historical open interest data with pagination."""
        exchange = self._get_exchange()
        try:
            if not hasattr(exchange, "fetch_open_interest_history"):
                return pd.DataFrame()

            max_days = self._OI_MAX_DAYS.get(timeframe, 30)
            effective_days = min(days_back, max_days)
            if effective_days < days_back:
                logger.info(
                    "OI history for %s capped to %d days (exchange limit for %s)",
                    symbol, max_days, timeframe,
                )

            since_ms = int(
                (
                    datetime.now(tz=timezone.utc) - timedelta(days=effective_days)
                ).timestamp()
                * 1000
            )
            end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

            all_records: list[dict] = []
            cursor = since_ms
            limit = 500

            while cursor < end_ms:
                oi_data = exchange.fetch_open_interest_history(
                    symbol, timeframe=timeframe, since=cursor, limit=limit,
                )
                if not oi_data:
                    break
                all_records.extend(oi_data)
                cursor = oi_data[-1]["timestamp"] + 1
                if len(oi_data) < limit:
                    break
                time.sleep(exchange.rateLimit / 1000)

            if all_records:
                return pd.DataFrame(
                    [
                        {
                            "timestamp": item.get("timestamp", 0),
                            "openInterest": item.get(
                                "openInterestValue",
                                item.get("openInterest", 0),
                            ),
                        }
                        for item in all_records
                    ]
                )
        except Exception as e:
            logger.warning("Error fetching open interest for %s: %s", symbol, e)
        return pd.DataFrame()
