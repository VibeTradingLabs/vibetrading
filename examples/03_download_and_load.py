"""
Example 3: Data download and load workflow.

Demonstrates:
  1. vibetrading.tools.download_data() fetches OHLCV + funding rate via CCXT
  2. vibetrading.tools.load_csv() and helpers read the cache back

The downloaded data is cached to CSV files so subsequent runs skip the download.

If the exchange API is unreachable (e.g. behind a firewall), pass the
``proxy`` parameter or set the ``HTTPS_PROXY`` environment variable::

    export HTTPS_PROXY=http://127.0.0.1:7890
    python examples/03_download_and_load.py

Usage:
    python examples/03_download_and_load.py
"""
import os
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

import vibetrading.tools

load_dotenv()

def main():
    assets = ["BTC", "ETH"]
    exchange = "binance"
    interval = "1h"

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=30)

    # Step 1: Download data for multiple assets
    print("=" * 60)
    print("Step 1: Downloading historical data via CCXT")
    print("=" * 60)
    data = vibetrading.tools.download_data(
        assets,
        exchange=exchange,
        start_time=start,
        end_time=end,
        interval=interval,
        market_type="perp",
        # proxy=os.environ["HTTPS_PROXY"], // if you need proxy
    )

    for key, df in data.items():
        if df.empty:
            print(f"  {key}: empty (download failed or no data)")
        else:
            print(f"  {key}: {len(df)} rows, "
                  f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")

    # Step 2: Load cached data back using data_loader
    print(f"\n{'=' * 60}")
    print("Step 2: Loading cached data via vibetrading.tools")
    print("=" * 60)

    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    for asset in assets:
        symbol = vibetrading.tools.DEFAULT_PERP_SYMBOLS.get(asset, f"{asset}/USDT:USDT")

        path = vibetrading.tools.generate_cache_filename(
            exchange=exchange,
            symbol=symbol,
            start_date=start_str,
            end_date=end_str,
            timeframe=interval,
        )
        print(f"\n  [{asset}] Cache file: {path}")

        df = vibetrading.tools.load_csv(path)
        if df.empty:
            print(f"  [{asset}] No data found in cache.")
            continue

        print(f"  [{asset}] Loaded {len(df)} rows")
        print(f"  [{asset}] Columns: {list(df.columns)}")
        print(f"  [{asset}] Date range: {df.index.min()} -> {df.index.max()}")
        print(f"  [{asset}] Latest close: {df['close'].iloc[-1]:.2f}")


if __name__ == "__main__":
    main()
