# VibeTrading

[![PyPI version](https://img.shields.io/pypi/v/vibetrading.svg)](https://pypi.org/project/vibetrading/)
[![CI](https://github.com/VibeTradingLabs/vibetrading/actions/workflows/ci.yml/badge.svg)](https://github.com/VibeTradingLabs/vibetrading/actions)
[![Python >= 3.10](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://pypi.org/project/vibetrading/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Agent-first trading framework for cryptocurrency. Describe strategies in natural language, generate executable Python code, backtest on historical data, and analyze with LLM-powered insights.

## Installation

```bash
pip install vibetrading
```

With exchange-specific extras:

```bash
pip install "vibetrading[hyperliquid]"  # Hyperliquid live trading
pip install "vibetrading[all]"          # All exchange adapters
pip install "vibetrading[dev]"          # Development tools (pytest, ruff)
```

## Quick Start

### 1. Generate a Strategy

Set your LLM API key:

```bash
export ANTHROPIC_API_KEY=sk-...  # or OPENAI_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY
```

```python
import vibetrading.strategy

code = vibetrading.strategy.generate(
    "BTC momentum strategy: RSI(14) oversold entry, SMA crossover confirmation, "
    "3x leverage, 10% position size, 8% take-profit, 4% stop-loss",
    model="claude-sonnet-4-20250514",
)
```

Or use a **built-in template** — no LLM needed:

```python
from vibetrading.templates import momentum

code = momentum.generate(asset="BTC", leverage=3, sma_fast=10, sma_slow=30)
```

### 2. Validate

```python
result = vibetrading.strategy.validate(code)
print(result)  # StrategyValidationResult(VALID, errors=0, warnings=2)
```

### 3. Backtest

```python
import vibetrading.backtest
import vibetrading.tools

data = vibetrading.tools.download_data(["BTC"], exchange="binance", interval="1h")

results = vibetrading.backtest.run(
    code,
    interval="1h",
    data=data,
    slippage_bps=5,  # realistic 0.05% slippage
)

metrics = results["metrics"]
print(f"Return:        {metrics['total_return']:.2%}")
print(f"CAGR:          {metrics['cagr']:.2%}")
print(f"Sharpe:        {metrics['sharpe_ratio']:.2f}")
print(f"Sortino:       {metrics['sortino_ratio']:.2f}")
print(f"Calmar:        {metrics['calmar_ratio']:.2f}")
print(f"Max Drawdown:  {metrics['max_drawdown']:.2%}")
print(f"Win Rate:      {metrics['win_rate']:.2%}")
print(f"Profit Factor: {metrics['profit_factor']:.2f}")
print(f"Expectancy:    ${metrics['expectancy']:.2f}/trade")
```

### 4. Analyze with LLM

```python
report = vibetrading.strategy.analyze(results, strategy_code=code, model="claude-sonnet-4-20250514")

print(f"Score: {report.score}/10")
print(report.summary)
for s in report.suggestions:
    print(f"  → {s}")
```

## CLI

Run everything from the terminal:

```bash
# Generate from template
vibetrading template --list
vibetrading template momentum asset=ETH leverage=5 -o strategy.py

# Validate
vibetrading validate strategy.py

# Download data
vibetrading download BTC ETH --exchange binance --interval 1h

# Backtest
vibetrading backtest strategy.py --interval 1h --balance 10000

# JSON output for pipelines
vibetrading backtest strategy.py --json
```

## How It Works

```
Describe  ──▶  Generate  ──▶  Validate  ──▶  Backtest  ──▶  Analyze
(prompt)       (LLM)          (static)       (engine)       (LLM)
```

1. **Describe** — Write what you want in plain English.
2. **Generate** — An LLM produces framework-compatible strategy code with risk management.
3. **Validate** — Static analysis catches common errors before execution.
4. **Backtest** — Run against historical data with realistic simulation.
5. **Analyze** — An LLM evaluates results: scores performance, finds weaknesses, suggests fixes.

## Features

### Strategy Generation
- **Any LLM** — OpenAI, Anthropic, Google, DeepSeek, or any provider via [litellm](https://github.com/BerriAI/litellm)
- **Built-in templates** — Momentum, mean reversion, grid, DCA — ready to use, no LLM required
- **Static validator** — Catches missing imports, leverage, risk management issues before runtime
- **LLM analysis** — Structured scoring (1–10), strengths/weaknesses, actionable suggestions

### Backtesting
- **Realistic simulation** — Market/limit orders, margin, leverage, funding rates, fees, liquidation detection
- **Slippage modeling** — Configurable basis-point slippage on market orders
- **Multi-exchange data** — Download and cache OHLCV from Binance, Bybit, OKX, and 100+ CCXT exchanges

### Comprehensive Metrics
- **Returns**: Total return, CAGR
- **Risk-adjusted**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Drawdown**: Max drawdown, max drawdown duration
- **Trade stats**: Win rate, profit factor, expectancy, avg win/loss, largest win/loss
- **Streaks**: Max consecutive wins/losses
- **Costs**: Transaction fees, funding revenue

### Live Trading
- **Exchange adapters**: Hyperliquid, Paradex, Lighter, Aster, Extended
- **Unified interface** — Same code runs in backtest and live via `VibeSandboxBase`
- **LiveRunner** — Periodic strategy execution with real exchange connections

### Developer Experience
- **CLI tool** — Generate, validate, backtest, and download from the terminal
- **`py.typed`** — Full PEP 561 typing support
- **CI/CD** — GitHub Actions with lint + test on Python 3.10/3.11/3.12
- **200+ tests** — Comprehensive coverage of all core modules

## Strategy Templates

Generate battle-tested strategies instantly:

```python
from vibetrading.templates import momentum, mean_reversion, grid, dca

# SMA crossover + RSI momentum
code = momentum.generate(asset="BTC", leverage=3, tp_pct=0.05, sl_pct=0.02)

# Bollinger Band mean reversion
code = mean_reversion.generate(asset="ETH", bb_period=20, bb_std=2.0)

# Spot grid trading
code = grid.generate(asset="SOL", grid_levels=10, grid_spacing_pct=0.008)

# Dollar-cost averaging
code = dca.generate(asset="BTC", buy_amount=100, interval="1d")
```

All templates are fully parameterizable and pass static validation.

## Built-in Indicators

Pure-pandas implementations — no `ta` library dependency:

```python
from vibetrading.indicators import rsi, sma, ema, bbands, atr, macd, stochastic, vwap

ohlcv = get_futures_ohlcv("BTC", "1h", 50)
rsi_14 = rsi(ohlcv["close"])
upper, middle, lower = bbands(ohlcv["close"])
macd_line, signal, hist = macd(ohlcv["close"])
atr_14 = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"])
k, d = stochastic(ohlcv["high"], ohlcv["low"], ohlcv["close"])
```

## Position Sizing

Built-in sizing methods for systematic risk management:

```python
from vibetrading.sizing import kelly_size, fixed_fraction_size, risk_per_trade_size

# Kelly Criterion (half-Kelly default for reduced variance)
qty = kelly_size(win_rate=0.55, avg_win=200, avg_loss=100, balance=10000, price=50000)

# Fixed risk per trade based on stop-loss distance
qty = risk_per_trade_size(balance=10000, risk_pct=0.01, entry=50000, stop_loss=49000)
```

## Modules

| Module | Purpose |
|---|---|
| `vibetrading` | `vibe` decorator |
| `vibetrading.strategy` | `generate()`, `validate()`, `analyze()`, prompt templates |
| `vibetrading.backtest` | `BacktestEngine`, `run()`, `StaticSandbox` |
| `vibetrading.tools` | `download_data()`, `load_csv()` |
| `vibetrading.templates` | `momentum`, `mean_reversion`, `grid`, `dca`, `multi_momentum` |
| `vibetrading.indicators` | `sma`, `ema`, `rsi`, `bbands`, `atr`, `macd`, `stochastic`, `vwap` |
| `vibetrading.sizing` | `kelly_size`, `fixed_fraction_size`, `volatility_adjusted_size`, `risk_per_trade_size` |
| `vibetrading.cli` | Command-line interface |

## Examples

See the [`examples/`](examples/) directory for complete working strategies:

1. Basic strategy generation
2. Backtest with data download
3. LLM-powered analysis
4. Custom strategy with technical indicators
5. Multi-asset portfolio
6. Live trading setup

## Contributing

```bash
git clone https://github.com/VibeTradingLabs/vibetrading.git
cd vibetrading
pip install -e ".[dev]"
pytest
```

## License

MIT
