# VibeTrading

[![PyPI version](https://img.shields.io/pypi/v/vibetrading.svg)](https://pypi.org/project/vibetrading/)
[![Python >= 3.10](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://pypi.org/project/vibetrading/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Agent-first trading framework for cryptocurrency. Describe strategies in natural language, generate executable Python code, backtest on historical data, and iteratively improve with LLM-powered analysis.

## Installation

```bash
pip install vibetrading
```

## Quick Start

### Setup

Set your LLM API key (at least one) and optional proxy:

```bash
export OPENAI_API_KEY=sk-...        # or ANTHROPIC_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY
export HTTPS_PROXY=http://127.0.0.1:7890   # optional
```

Or put them in a `.env` file — the package loads it automatically.

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
- **Built-in validator** — Static analysis catches common errors before execution.
- **LLM backtest analysis** — Structured scoring (1-10), strengths/weaknesses, and actionable suggestions.
- **Strategy evolution** — `vibetrading.evolve()` runs the full generate → backtest → analyze → regenerate loop.
- **CCXT data** — Download and cache OHLCV data from Binance, Bybit, OKX, and 100+ exchanges.
- **Realistic simulation** — Limit/market orders, margin, leverage, funding rates, fees, and liquidation detection.

## Modules

| Module | Purpose |
|---|---|
| `vibetrading` | `vibe` decorator, `evolve()` |
| `vibetrading.strategy` | `generate()`, `validate()`, `analyze()`, prompt templates |
| `vibetrading.backtest` | `BacktestEngine`, `run()` |
| `vibetrading.evolution` | `StrategyEvolver`, `evolve()` |
| `vibetrading.tools` | `download_data()`, `load_csv()` |

## License

MIT
