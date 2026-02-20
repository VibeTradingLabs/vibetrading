# VibeTrading

Describe trading strategies in natural language. Get executable Python. Backtest and deploy to any exchange.

```bash
pip install vibetrading
```

---

## How It Works

**1. Describe** — Tell the agent what you want in plain English.

**2. Generate** — AI produces framework-compatible strategy code with proper risk management.

**3. Download & Backtest** — Fetch historical data with the CCXT downloader tool, then backtest. Deploy to a live exchange with the same code.

---

## Quick Start

### Generate a Strategy from a Prompt

```python
from vibetrading import StrategyGenerator

generator = StrategyGenerator(model="gpt-4o")

code = generator.generate(
    "BTC momentum strategy: RSI(14) oversold entry, SMA crossover confirmation, "
    "3x leverage, 10% position size, 8% take-profit, 4% stop-loss",
    assets=["BTC"],
    max_leverage=5,
)

print(code)
```

### Generate and Backtest

```python
from datetime import datetime, timezone
from vibetrading import StrategyGenerator, BacktestEngine
from vibetrading.tools import download_data

start = datetime(2025, 1, 1, tzinfo=timezone.utc)
end = datetime(2025, 6, 1, tzinfo=timezone.utc)

# Step 1: Generate strategy code
generator = StrategyGenerator(model="gpt-4o")
code = generator.generate(
    "ETH mean reversion with Bollinger Bands, short when price hits upper band, "
    "long when price hits lower band, 5x leverage",
    assets=["ETH"],
    max_leverage=5,
)

# Step 2: Download historical data
data = download_data(
    ["ETH"],
    exchange="binance",
    start_time=start,
    end_time=end,
    interval="1h",
)

# Step 3: Backtest
engine = BacktestEngine(
    start_time=start,
    end_time=end,
    interval="1h",
    exchange="binance",
    initial_balances={"USDC": 10000},
    data=data,
)

results = engine.run(code)

if results:
    metrics = results["metrics"]
    print(f"Return: {metrics['total_return']:.2%}")
    print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
```

### Use the Prompt Template with Any LLM

Don't want to use the built-in generator? Use the prompt template directly with any LLM client:

```python
import openai
from vibetrading.agent import build_generation_prompt

messages = build_generation_prompt(
    "BTC grid strategy with 0.25% spacing, 72 levels per side, 5x leverage",
    assets=["BTC"],
    market_type="perp",
    max_leverage=5,
)

response = openai.chat.completions.create(model="gpt-4o", messages=messages)
strategy_code = response.choices[0].message.content
```

Or with Anthropic:

```python
import anthropic
from vibetrading.agent import STRATEGY_SYSTEM_PROMPT, build_generation_prompt

messages = build_generation_prompt("SOL scalping with VWAP and RSI")

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    system=messages[0]["content"],
    messages=[{"role": "user", "content": messages[1]["content"]}],
)
strategy_code = response.content[0].text
```

### Validate Generated Code

Check generated strategy code for common errors before running:

```python
from vibetrading import validate_strategy

result = validate_strategy(strategy_code)

if result.is_valid:
    print("Strategy passed validation")
else:
    print(result)
    # Feed errors back to LLM for correction
    feedback = result.format_for_llm()
```

---

## Write Strategies Manually

You can also write strategies by hand. A strategy is a Python function decorated with `@vibe`:

```python
import math
import ta
from vibetrading import (
    vibe,
    get_current_time,
    get_perp_price,
    get_futures_ohlcv,
    get_perp_summary,
    get_perp_position,
    long,
    reduce_position,
    set_leverage,
)

ASSET = "BTC"
LEVERAGE = 3
TP_PCT = 0.08
SL_PCT = 0.04
RISK_PER_TRADE_PCT = 0.10
RSI_OVERSOLD = 30
SMA_FAST = 10
SMA_SLOW = 20


@vibe(interval="1m")
def my_strategy():
    current_price = get_perp_price(ASSET)
    if math.isnan(current_price):
        return

    perp_summary = get_perp_summary()
    available_margin = perp_summary.get("available_margin", 0.0)
    position = get_perp_position(ASSET)

    # Risk management (every frame)
    if position:
        size = position.get("size", 0.0)
        entry_price = position.get("entry_price", 0.0)
        pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0

        if pnl_pct >= TP_PCT:
            reduce_position(ASSET, abs(size) * 0.5)
            return
        elif pnl_pct <= -SL_PCT:
            reduce_position(ASSET, abs(size))
            return
        return

    # Entry logic (only when flat)
    ohlcv = get_futures_ohlcv(ASSET, "1m", SMA_SLOW + 10)
    if len(ohlcv) < SMA_SLOW:
        return

    rsi = ta.momentum.rsi(ohlcv["close"], window=14).iloc[-1]
    sma_fast = ohlcv["close"].rolling(SMA_FAST).mean().iloc[-1]
    sma_slow = ohlcv["close"].rolling(SMA_SLOW).mean().iloc[-1]

    if rsi < RSI_OVERSOLD and sma_fast > sma_slow:
        set_leverage(ASSET, LEVERAGE)
        qty = (available_margin * RISK_PER_TRADE_PCT * LEVERAGE) / current_price
        long(ASSET, qty, price=current_price)
```

### Backtest

```python
from datetime import datetime, timezone
from vibetrading import BacktestEngine
from vibetrading.tools import download_data

start = datetime(2025, 1, 1, tzinfo=timezone.utc)
end = datetime(2025, 7, 1, tzinfo=timezone.utc)

# Step 1: Download historical data
data = download_data(
    ["BTC"],
    exchange="binance",
    start_time=start,
    end_time=end,
    interval="1h",
)

# Step 2: Run backtest with pre-downloaded data
engine = BacktestEngine(
    start_time=start,
    end_time=end,
    interval="1h",
    exchange="binance",
    initial_balances={"USDC": 10000},
    data=data,
)

results = engine.run(strategy_code)

print(results["metrics"])
# {
#   "total_return": 0.127,
#   "max_drawdown": -0.054,
#   "sharpe_ratio": 1.82,
#   "win_rate": 0.61,
#   "number_of_trades": 48,
#   ...
# }
```

### Go Live

Same strategy code. Same API. Different runtime.

```python
import asyncio
from vibetrading import create_sandbox, LiveRunner

sandbox = create_sandbox(
    "hyperliquid",
    api_key="0xYourWalletAddress",
    api_secret="0xYourPrivateKey",
)

runner = LiveRunner(sandbox, interval="1m")
runner.load_strategy(strategy_code)
asyncio.run(runner.start())
```

---

## Core Concepts

### The `@vibe` Decorator

Every strategy must have exactly ONE function decorated with `@vibe`. This registers the function as a callback that the engine executes at each tick.

```python
from vibetrading import vibe

@vibe(interval="1m")
def on_tick():
    pass
```

For live trading, always use `interval="1m"`. Implement frame-skipping for longer intervals:

```python
last_execution_time = None

@vibe(interval="1m")
def strategy():
    global last_execution_time
    current_time = get_current_time()

    # Risk management runs every frame
    manage_risk()

    # Main logic every 5 minutes
    if last_execution_time and (current_time - last_execution_time).total_seconds() < 300:
        return
    last_execution_time = current_time
    # ... main logic ...
```

### The Sandbox Interface

All trading operations go through a unified interface (`VibeSandboxBase`). Whether you are backtesting or live trading, the API is identical:

| Category | Functions |
|---|---|
| **Account** | `get_spot_summary()`, `get_perp_summary()`, `get_perp_position(asset)` |
| **Trading** | `buy(asset, qty, price)`, `sell(asset, qty, price)` |
| **Futures** | `long(asset, qty, price)`, `short(asset, qty, price)`, `reduce_position(asset, qty)` |
| **Leverage** | `set_leverage(asset, leverage)` |
| **Price** | `get_perp_price(asset)`, `get_spot_price(asset)` |
| **OHLCV** | `get_spot_ohlcv(asset, interval, limit)`, `get_futures_ohlcv(asset, interval, limit)` |
| **Funding** | `get_funding_rate(asset)`, `get_funding_rate_history(asset, limit)` |
| **OI** | `get_open_interest(asset)`, `get_open_interest_history(asset, limit)` |
| **Orders** | `get_perp_open_orders()`, `get_spot_open_orders()`, `cancel_perp_orders(asset, ids)` |
| **Time** | `get_current_time()` |

### Architecture

```
User Prompt (natural language)
         │
         ▼
┌─────────────────────────┐
│   LLM Agent             │  ← any model (GPT, Claude, Gemini, ...)
│   + prompt template     │  ← STRATEGY_SYSTEM_PROMPT
└────────┬────────────────┘
         │ generates
         ▼
Strategy Code (@vibe decorated)
         │
         ▼
┌─────────────────────────┐
│   vibetrading module    │  ← runtime-injected API
│   (mock namespace)      │
└────────┬────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
Backtest    Live
Engine      Runner
    │         │
    ▼         ▼
Static     Exchange
Sandbox    Sandbox
    │      (Hyperliquid, Paradex,
    ▼       Extended, Lighter, ...)
tools/data_downloader
(CCXT → CSV cache)
```

---

## Installation

### Basic (backtesting only)

```bash
pip install vibetrading
```

### With strategy generation

```bash
pip install "vibetrading[agent]"
```

Installs `litellm` for multi-provider LLM support (OpenAI, Anthropic, Google, etc.).

### With exchange support

```bash
# Hyperliquid
pip install "vibetrading[hyperliquid]"

# X10 Extended (StarkNet)
pip install "vibetrading[extended]"

# Paradex (StarkNet)
pip install "vibetrading[paradex]"

# Lighter (zkSync Era)
pip install "vibetrading[lighter]"

# Aster Protocol
pip install "vibetrading[aster]"

# Everything
pip install "vibetrading[all]"
```

### With technical analysis

```bash
pip install "vibetrading[ta]"
```

The `ta` library is auto-detected at runtime. If installed, `import ta` works inside strategy code.

---

## Agent Integration

### Using VibeTrading as an Agent Skill

The structured `@vibe` interface makes VibeTrading a composable skill for autonomous agent systems. The key components:

| Component | Import | Purpose |
|---|---|---|
| `STRATEGY_SYSTEM_PROMPT` | `from vibetrading.agent import STRATEGY_SYSTEM_PROMPT` | Complete system prompt for LLM strategy generation |
| `VIBETRADING_API_REFERENCE` | `from vibetrading.agent import VIBETRADING_API_REFERENCE` | API documentation string |
| `STRATEGY_CONSTRAINTS` | `from vibetrading.agent import STRATEGY_CONSTRAINTS` | Code generation rules |
| `build_generation_prompt()` | `from vibetrading.agent import build_generation_prompt` | Build message list for chat completion |
| `validate_strategy()` | `from vibetrading import validate_strategy` | Validate generated code |
| `StrategyGenerator` | `from vibetrading import StrategyGenerator` | Full generation + validation pipeline |

### Closed-Loop Generation

The validator produces structured feedback that can be fed back to the LLM:

```python
from vibetrading import StrategyGenerator, validate_strategy
from vibetrading.agent import build_generation_prompt

messages = build_generation_prompt("BTC scalping strategy with VWAP")

# First attempt
code = call_your_llm(messages)
result = validate_strategy(code)

if not result.is_valid:
    # Feed errors back
    messages.append({"role": "assistant", "content": code})
    messages.append({"role": "user", "content": result.format_for_llm()})

    # Retry
    code = call_your_llm(messages)
```

Or use `StrategyGenerator` which handles this automatically:

```python
generator = StrategyGenerator(model="gpt-4o")
code = generator.generate("BTC scalping", validate=True, max_retries=3)
```

---

## Backtesting Guide

### Backtest Results

`engine.run()` returns a dictionary containing:

```python
results["metrics"]          # Performance metrics dict
results["trades"]           # List of all executed trades
results["final_balances"]   # Final asset balances
results["results"]          # Time-series DataFrame of portfolio values
results["simulation_info"]  # Metadata (steps, time range, liquidation status)
```

### Metrics Included

| Metric | Description |
|---|---|
| `total_return` | Total portfolio return (decimal) |
| `max_drawdown` | Maximum peak-to-trough drawdown |
| `sharpe_ratio` | Annualized Sharpe ratio |
| `win_rate` | Percentage of profitable closed trades |
| `number_of_trades` | Total number of trades executed |
| `funding_revenue` | Net funding payments received/paid |
| `total_tx_fees` | Total transaction fees paid |
| `average_trade_duration_hours` | Mean holding period |

### Supported Intervals

`1s`, `1m`, `5m`, `15m`, `30m`, `1h`, `6h`, `1d`

### Supported Exchanges for Backtesting

Data is fetched from exchanges via CCXT. Download data first, then pass it to the backtest engine:

```python
from vibetrading.tools import download_data

# Download from any CCXT-supported exchange
data = download_data(["BTC", "ETH"], exchange="binance", ...)
data = download_data(["BTC"], exchange="bybit", ...)
data = download_data(["BTC"], exchange="okx", ...)

# Pass to BacktestEngine
engine = BacktestEngine(exchange="binance", data=data, ...)
```

---

## Live Trading Guide

### Step 1: Create a Sandbox

```python
from vibetrading import create_sandbox

sandbox = create_sandbox(
    "hyperliquid",
    api_key="0xYourWalletAddress",
    api_secret="0xYourPrivateKey",
)
```

### Step 2: Load Strategy

```python
from vibetrading import LiveRunner

runner = LiveRunner(sandbox, interval="1m")
runner.load_strategy(strategy_code)
```

### Step 3: Run

```python
import asyncio
asyncio.run(runner.start())
```

### Step 4: Run a Single Iteration (for testing)

```python
runner.load_strategy(strategy_code)
runner.run_callbacks_once()
runner.cleanup()
```

---

## Supported Exchanges

| Exchange | Type | Status | Install |
|---|---|---|---|
| Hyperliquid | Perps + Spot | Full implementation | `vibetrading[hyperliquid]` |
| X10 Extended | Perps | Adapter ready | `vibetrading[extended]` |
| Paradex | Perps | Adapter ready | `vibetrading[paradex]` |
| Lighter | Perps + Spot | Adapter ready | `vibetrading[lighter]` |
| Aster | Perps | Adapter ready | `vibetrading[aster]` |

### Adding a Custom Exchange

Implement the `VibeSandboxBase` interface:

```python
from vibetrading.core.sandbox_base import VibeSandboxBase

class MyExchangeSandbox(VibeSandboxBase):
    def get_price(self, asset: str) -> float:
        ...

    def long(self, asset, quantity, price, order_type="limit"):
        ...

    # ... implement all abstract methods
```

---

## Configuration

Environment variables (optional):

| Variable | Description | Default |
|---|---|---|
| `VIBETRADING_DEFAULT_EXCHANGE` | Default exchange for data downloads | `binance` |
| `{EXCHANGE}_API_KEY` | Per-exchange API key (e.g. `BINANCE_API_KEY`) | `None` |
| `{EXCHANGE}_API_SECRET` | Per-exchange API secret (e.g. `BINANCE_API_SECRET`) | `None` |
| `{EXCHANGE}_PASSWORD` | Per-exchange passphrase (OKX, KuCoin, etc.) | `None` |
| *(removed)* | Dataset directory is always `<cwd>/vibetrading/dataset` | — |

Exchange credentials can also be set programmatically (CCXT-compatible dict):

```python
from vibetrading.config import EXCHANGES

EXCHANGES["binance"] = {"apiKey": "...", "secret": "..."}
EXCHANGES["okx"] = {"apiKey": "...", "secret": "...", "password": "..."}
```

---

## Project Structure

```
vibetrading/
├── __init__.py              # Public API
├── config.py                # Configuration
├── agent/
│   ├── prompt.py            # System prompt, API reference, constraints
│   ├── generator.py         # StrategyGenerator + generate_strategy()
│   └── validator.py         # validate_strategy()
├── core/
│   ├── sandbox_base.py      # VibeSandboxBase (abstract interface)
│   ├── decorator.py         # @vibe decorator
│   ├── error_handler.py     # Strategy error capture
│   ├── static_sandbox.py    # Backtesting sandbox
│   ├── backtest.py          # BacktestEngine
│   └── live_runner.py       # LiveRunner
├── exchanges/
│   ├── base.py              # LiveSandboxBase
│   ├── hyperliquid.py       # Hyperliquid adapter
│   ├── extended.py          # X10 Extended adapter
│   ├── paradex.py           # Paradex adapter
│   ├── lighter.py           # Lighter adapter
│   └── aster.py             # Aster adapter
├── models/
│   ├── orders.py            # Order & position models
│   └── types.py             # Market metadata & enums
├── metrics/
│   └── calculator.py        # Performance metrics
├── tools/
│   ├── data_downloader.py   # download_data() + CCXT data fetching
│   └── data_loader.py       # CSV cache loading + symbol mappings
└── utils/
    ├── math.py              # Numeric precision
    ├── json.py              # Serialization
    ├── cache.py             # API call caching
    ├── notification.py      # Error deduplication
    └── logging.py           # Structured logging
```

---

## Requirements

- Python >= 3.10
- pandas >= 2.0
- numpy >= 1.24
- pydantic >= 2.0
- ccxt >= 4.0
- litellm >= 1.0 (optional, for strategy generation)
- ta >= 0.11 (optional, for technical analysis indicators)

---

## License

MIT
