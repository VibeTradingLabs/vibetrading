# VibeTrading

[![PyPI version](https://img.shields.io/pypi/v/vibetrading.svg)](https://pypi.org/project/vibetrading/)
[![Python](https://img.shields.io/pypi/pyversions/vibetrading.svg)](https://pypi.org/project/vibetrading/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Agent-first trading framework for cryptocurrency. Describe strategies in natural language, generate executable Python code, backtest on historical data, and iteratively improve with LLM-powered analysis.

## Installation

```bash
pip install vibetrading
```

All core dependencies are included: `pandas`, `numpy`, `ccxt`, `litellm`, `ta`, `pydantic`, `python-dotenv`.

## Quick Start

### Setup

Set your LLM API key (at least one):

```bash
export OPENAI_API_KEY=sk-...
# or
export ANTHROPIC_API_KEY=sk-ant-...
# or
export GEMINI_API_KEY=...
# or
export DEEPSEEK_API_KEY=...
```

If you need an HTTP proxy:

```bash
export HTTPS_PROXY=http://127.0.0.1:7890
```

Or put them in a `.env` file — the package loads it automatically via `python-dotenv`.

### Generate a Strategy

```python
import vibetrading.strategy

code = vibetrading.strategy.generate(
    "BTC momentum strategy: RSI(14) oversold entry, SMA crossover confirmation, "
    "3x leverage, 10% position size, 8% take-profit, 4% stop-loss",
    model="gpt-4o",
)
```

### Backtest

```python
import vibetrading.backtest
import vibetrading.tools

data = vibetrading.tools.download_data(["BTC"], exchange="binance", interval="1h")

results = vibetrading.backtest.run(code, interval="1h", data=data)

metrics = results["metrics"]
print(f"Return: {metrics['total_return']:.2%}")
print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"Win Rate: {metrics['win_rate']:.2%}")
```

### Analyze Results

Use an LLM to score performance and get actionable improvement suggestions:

```python
report = vibetrading.strategy.analyze(results, strategy_code=code, model="gpt-4o")

print(f"Score: {report.score}/10")
print(report.summary)
for s in report.suggestions:
    print(f"  → {s}")
```

### Evolve

One call to iteratively improve a strategy through generate → backtest → analyze feedback loops:

```python
import vibetrading

result = vibetrading.evolve(
    "BTC momentum strategy with RSI and SMA crossover, 3x leverage",
    iterations=3,
    model="gpt-4o",
    interval="1h",
)

print(f"Best score: {result.best_score}/10")
print(f"Improved: {result.improved}")
print(result.best_code)
```

## How It Works

```
Describe  ──▶  Generate  ──▶  Backtest  ──▶  Analyze  ──▶  Evolve
(prompt)       (LLM)          (engine)       (LLM)         (loop)
                  ▲                                           │
                  └──────────── feedback ──────────────────────┘
```

1. **Describe** — Write what you want in plain English.
2. **Generate** — An LLM produces framework-compatible strategy code with risk management.
3. **Backtest** — Run against historical data from any CCXT-supported exchange.
4. **Analyze** — An LLM evaluates backtest results: scores performance, finds weaknesses, suggests fixes.
5. **Evolve** — Repeat the loop. Each iteration feeds analysis back to the generator.

## Features

- **Any LLM** — Works with OpenAI, Anthropic, Google, DeepSeek, or any OpenAI-compatible API via [litellm](https://github.com/BerriAI/litellm).
- **Built-in validator** — Static analysis catches common errors (missing `@vibe`, no `set_leverage`, hardcoded balances, etc.) before execution.
- **LLM backtest analysis** — Structured scoring (1-10), strengths/weaknesses, and improvement suggestions with `format_for_llm()` for closed-loop feedback.
- **Strategy evolution** — `vibetrading.evolve()` runs the full generate → backtest → analyze → regenerate loop in one call.
- **CCXT data** — Download and cache OHLCV data from Binance, Bybit, OKX, and 100+ exchanges.
- **Realistic simulation** — Limit/market orders, margin, leverage, funding rates, fees, and liquidation detection.
- **`ta` integration** — `import ta` works inside strategy code when installed.

## Modules

| Module | Purpose |
|---|---|
| `vibetrading` | `vibe` decorator, `evolve()` |
| `vibetrading.strategy` | `generate()`, `validate()`, `analyze()`, prompt templates |
| `vibetrading.backtest` | `BacktestEngine`, `run()` |
| `vibetrading.evolution` | `StrategyEvolver`, `evolve()` |
| `vibetrading.tools` | `download_data()`, `load_csv()` |
| `vibetrading.models` | Order & position data models |

## Strategy API

All functions are available inside `@vibe`-decorated strategy code via `from vibetrading import ...`:

| Category | Functions |
|---|---|
| **Account** | `get_spot_summary()`, `get_perp_summary()`, `get_perp_position(asset)` |
| **Spot** | `buy(asset, qty, price)`, `sell(asset, qty, price)` |
| **Futures** | `long(asset, qty, price)`, `short(asset, qty, price)`, `reduce_position(asset, qty)` |
| **Leverage** | `set_leverage(asset, leverage)` |
| **Price** | `get_perp_price(asset)`, `get_spot_price(asset)` |
| **OHLCV** | `get_spot_ohlcv(asset, interval, limit)`, `get_futures_ohlcv(asset, interval, limit)` |
| **Funding** | `get_funding_rate(asset)`, `get_funding_rate_history(asset, limit)` |
| **OI** | `get_open_interest(asset)`, `get_open_interest_history(asset, limit)` |
| **Orders** | `get_perp_open_orders()`, `get_spot_open_orders()`, `cancel_perp_orders(asset, ids)` |
| **Time** | `get_current_time()` |

## Backtest Metrics

| Metric | Description |
|---|---|
| `total_return` | Total portfolio return (decimal) |
| `max_drawdown` | Maximum peak-to-trough drawdown |
| `sharpe_ratio` | Annualized Sharpe ratio |
| `win_rate` | Percentage of profitable closed trades |
| `number_of_trades` | Total trades executed |
| `funding_revenue` | Net funding payments received/paid |
| `total_tx_fees` | Total transaction fees |
| `average_trade_duration_hours` | Mean holding period |

Supported intervals: `1s`, `1m`, `5m`, `15m`, `30m`, `1h`, `6h`, `1d`

## Writing Strategies by Hand

A strategy is a Python function decorated with `@vibe`:

```python
import math
from vibetrading import (
    vibe, get_perp_price, get_futures_ohlcv,
    get_perp_summary, get_perp_position,
    long, reduce_position, set_leverage,
)

ASSET = "BTC"
LEVERAGE = 3
TP_PCT, SL_PCT = 0.08, 0.04

@vibe(interval="1m")
def my_strategy():
    price = get_perp_price(ASSET)
    if math.isnan(price):
        return

    summary = get_perp_summary()
    margin = summary.get("available_margin", 0.0)
    position = get_perp_position(ASSET)

    if position:
        entry = position.get("entry_price", 0.0)
        size = position.get("size", 0.0)
        pnl_pct = (price - entry) / entry if entry > 0 else 0
        if pnl_pct >= TP_PCT:
            reduce_position(ASSET, abs(size) * 0.5)
        elif pnl_pct <= -SL_PCT:
            reduce_position(ASSET, abs(size))
        return

    ohlcv = get_futures_ohlcv(ASSET, "1h", 60)
    if len(ohlcv) < 50:
        return

    sma_fast = ohlcv["close"].rolling(10).mean().iloc[-1]
    sma_slow = ohlcv["close"].rolling(50).mean().iloc[-1]

    if sma_fast > sma_slow:
        set_leverage(ASSET, LEVERAGE)
        qty = (margin * 0.10 * LEVERAGE) / price
        if qty * price >= 15.0:
            long(ASSET, qty, price=price)
```

> Strategy code uses `from vibetrading import ...` — these symbols are injected at runtime by the backtest engine.

## Using Prompt Templates Directly

Use the prompt template with any LLM client instead of the built-in generator:

```python
import vibetrading.strategy

messages = vibetrading.strategy.build_generation_prompt(
    "BTC grid strategy with 0.25% spacing, 72 levels per side",
    assets=["BTC"],
    market_type="perp",
)

# Use with OpenAI, Anthropic, or any chat completion API
response = your_llm_client(messages)
```

Available prompt components:

| Export | Description |
|---|---|
| `STRATEGY_SYSTEM_PROMPT` | Complete system prompt with API reference and constraints |
| `VIBETRADING_API_REFERENCE` | Trading API documentation string |
| `STRATEGY_CONSTRAINTS` | Code generation rules and best practices |
| `build_generation_prompt()` | Build a message list for chat completion |

## Configuration

Copy `.env.dev_example` to `.env` and fill in the keys you need:

```bash
# HTTP proxy (optional)
HTTPS_PROXY=''

# LLM API keys — set at least one for strategy generation, analysis, and evolution
OPENAI_API_KEY=''
ANTHROPIC_API_KEY=''
GEMINI_API_KEY=''
DEEPSEEK_API_KEY=''
XAI_API_KEY=''
```

Only one LLM key is required. The package auto-detects which provider to use based on the `model` parameter passed to `generate()`, `analyze()`, or `evolve()`.

## Requirements

- Python >= 3.10
- pandas >= 2.0
- numpy >= 1.24
- pydantic >= 2.0
- python-dotenv >= 1.0
- ccxt >= 4.0
- litellm >= 1.80.0
- ta >= 0.11

## License

MIT
