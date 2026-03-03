# Live Trading Guide

Run your backtested strategies against real exchanges with real money.

> **⚠️ Warning:** Live trading involves real financial risk. Always test with small amounts first. Start with paper trading if available. vibetrading is not responsible for any losses.

## Supported Exchanges

| Exchange | Type | Auth | Install |
|---|---|---|---|
| [Hyperliquid](https://hyperliquid.xyz) | Perps + Spot | Wallet address + private key | `pip install "vibetrading[hyperliquid]"` |
| [Paradex](https://paradex.trade) | Perps | StarkNet key + account address | `pip install "vibetrading[paradex]"` |
| [Lighter](https://lighter.xyz) | Perps | API key + secret | `pip install "vibetrading[lighter]"` |
| [Aster](https://aster.finance) | Perps | API key + secret + user address | `pip install "vibetrading[aster]"` |

## Quick Start: Hyperliquid

Hyperliquid is the most popular exchange adapter. Here's how to go live:

### 1. Install

```bash
pip install "vibetrading[hyperliquid]"
```

### 2. Get Your Credentials

Hyperliquid uses Ethereum wallet authentication:

- **API Key** (`api_key`): Your Ethereum wallet address (e.g., `0x1234...abcd`)
- **API Secret** (`api_secret`): Your wallet's private key (e.g., `0xdead...beef`)

> **Security tip:** Use a dedicated trading wallet. Never use your main wallet. Create a new wallet, fund it with only what you're willing to trade, and use that private key.

You can also use Hyperliquid's [API wallet](https://app.hyperliquid.xyz/API) feature to create a sub-wallet with limited permissions.

### 3. Store Credentials Safely

Create a `.env.local` file (already gitignored):

```bash
# .env.local
HYPERLIQUID_WALLET=0xYourWalletAddress
HYPERLIQUID_PRIVATE_KEY=0xYourPrivateKey
```

### 4. Run Your Strategy

```python
import os
import asyncio
from dotenv import load_dotenv
import vibetrading.live

load_dotenv(".env.local")

strategy = open("strategies/rsi_mean_reversion.py").read()

asyncio.run(
    vibetrading.live.start(
        strategy,
        exchange="hyperliquid",
        api_key=os.environ["HYPERLIQUID_WALLET"],
        api_secret=os.environ["HYPERLIQUID_PRIVATE_KEY"],
        interval="1m",  # Execute every minute
    )
)
```

Or use the synchronous wrapper:

```python
vibetrading.live.start_sync(
    strategy,
    exchange="hyperliquid",
    api_key=os.environ["HYPERLIQUID_WALLET"],
    api_secret=os.environ["HYPERLIQUID_PRIVATE_KEY"],
    interval="1m",
)
```

### 5. Stop

Press `Ctrl+C` to stop the strategy.

## Advanced: Using the Sandbox Directly

For more control, use the sandbox and runner directly:

```python
import os
import asyncio
from dotenv import load_dotenv
import vibetrading.sandbox

load_dotenv(".env.local")

# Create the exchange sandbox
sandbox = vibetrading.sandbox.create(
    "hyperliquid",
    api_key=os.environ["HYPERLIQUID_WALLET"],
    api_secret=os.environ["HYPERLIQUID_PRIVATE_KEY"],
)

# Check connection
price = sandbox.get_perp_price("BTC")
print(f"BTC price: ${price:,.2f}")

# Check your balance
summary = sandbox.get_perp_summary()
print(f"Available margin: ${summary.available_margin:,.2f}")

# Load and run strategy
runner = vibetrading.sandbox.LiveRunner(sandbox, interval="1m")
runner.load_strategy(open("my_strategy.py").read())

# Run the loop
asyncio.run(runner.start())
```

## Exchange-Specific Setup

### Paradex (StarkNet)

```python
import vibetrading.live

asyncio.run(
    vibetrading.live.start(
        strategy,
        exchange="paradex",
        api_key="0xStarkNetPublicKey",
        api_secret="0xStarkNetPrivateKey",
        account_address="0xYourAccountAddress",
        interval="1m",
    )
)
```

### Lighter (zkSync Era)

```python
asyncio.run(
    vibetrading.live.start(
        strategy,
        exchange="lighter",
        api_key="your-api-key",
        api_secret="your-api-secret",
        interval="1m",
    )
)
```

### Aster Protocol

```python
asyncio.run(
    vibetrading.live.start(
        strategy,
        exchange="aster",
        api_key="your-api-key",
        api_secret="your-api-secret",
        user_address="0xYourAddress",
        interval="5m",
    )
)

# Testnet
asyncio.run(
    vibetrading.live.start(
        strategy,
        exchange="aster_testnet",
        api_key="your-api-key",
        api_secret="your-api-secret",
        user_address="0xYourAddress",
        interval="5m",
    )
)
```

## Execution Intervals

The `interval` parameter controls how often your strategy runs:

| Interval | Use Case |
|---|---|
| `"1s"` | High-frequency (use with caution — rate limits) |
| `"1m"` | Standard — good for most strategies |
| `"5m"` | Medium-frequency, lower API usage |
| `"15m"` | Swing trading |
| `"1h"` | Hourly rebalancing |
| `"1d"` | Daily rebalancing (DCA, etc.) |

> **Note:** The `@vibe(interval="1h")` decorator in your strategy code sets the *candle interval* for data lookups. The `LiveRunner` interval sets *how often the strategy function is called*. They can differ — e.g., your strategy might use `1h` candles but execute every `5m`.

## Same Code, Backtest and Live

The key design principle: **your strategy code is identical for backtesting and live trading.** The `@vibe` decorator, `get_perp_price()`, `long()`, etc. work in both contexts — the sandbox handles the difference.

```python
# This code works in both backtest and live:
from vibetrading import vibe, get_perp_price, get_perp_position, long, reduce_position
from vibetrading import get_perp_summary, set_leverage, get_futures_ohlcv
from vibetrading.indicators import rsi
import math

@vibe(interval="1h")
def my_strategy():
    price = get_perp_price("BTC")
    if math.isnan(price):
        return

    position = get_perp_position("BTC")
    if position and position.get("size", 0) != 0:
        pnl_pct = (price - position["entry_price"]) / position["entry_price"]
        if pnl_pct >= 0.03 or pnl_pct <= -0.02:
            reduce_position("BTC", abs(position["size"]))
        return

    ohlcv = get_futures_ohlcv("BTC", "1h", 20)
    if ohlcv is None or len(ohlcv) < 15:
        return

    if rsi(ohlcv["close"]).iloc[-1] < 30:
        summary = get_perp_summary()
        margin = summary.get("available_margin", 0)
        if margin > 100:
            set_leverage("BTC", 3)
            qty = (margin * 0.1 * 3) / price
            if qty * price >= 15:
                long("BTC", qty, price, order_type="market")
```

**Backtest it:**
```python
results = vibetrading.backtest.run(code, interval="1h", slippage_bps=5)
```

**Trade it live:**
```python
vibetrading.live.start_sync(code, exchange="hyperliquid", api_key=..., api_secret=...)
```

## Security Best Practices

1. **Never hardcode credentials** — Use environment variables or `.env.local`
2. **Never commit `.env.local`** — It's already in `.gitignore`
3. **Use a dedicated trading wallet** — Fund it with only what you're willing to lose
4. **Use API wallets** — Hyperliquid supports sub-wallets with limited permissions
5. **Start small** — Test with minimal amounts before scaling up
6. **Monitor your strategy** — Don't leave it running unattended until you trust it
7. **Set leverage conservatively** — 2-3x max for automated strategies
8. **Always have stop losses** — Never run a strategy without exit conditions

## Troubleshooting

### `ImportError: hyperliquid SDK not installed`

```bash
pip install "vibetrading[hyperliquid]"
```

### `ModuleNotFoundError: No module named 'vibetrading.indicators'`

Update to vibetrading >= 0.3.0. This was a sandbox bug fixed in v0.3.0.

### Rate limiting

If you're hitting rate limits, increase your execution interval:

```python
runner = vibetrading.sandbox.LiveRunner(sandbox, interval="5m")  # Instead of "1m"
```

### Strategy errors don't stop the runner

By design, the `LiveRunner` catches strategy errors and continues running. Check logs for error details:

```python
import logging
logging.basicConfig(level=logging.INFO)
```
