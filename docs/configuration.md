# Configuration

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | For AI generation | Anthropic API key for strategy generation |
| `OPENAI_API_KEY` | For AI generation | OpenAI API key (alternative to Anthropic) |

Store in `.env.local` at project root (gitignored by default).

## Backtest Configuration

### Slippage Modeling

Slippage simulates realistic market impact on market orders:

```python
results = vibetrading.backtest.run(
    code,
    slippage_bps=5,  # 5 basis points = 0.05%
)
```

- Only applies to market orders (limit orders fill at exact price)
- Buys fill slightly higher, sells fill slightly lower
- Typical values: 1-10 bps for liquid assets, 10-50 bps for illiquid

### Data Options

**Auto-download from Binance:**

```python
results = vibetrading.backtest.run(code, interval="1h")
```

**Pre-loaded data (faster, custom date ranges):**

```python
import vibetrading.tools

data = vibetrading.tools.download_data(
    ["BTC", "ETH", "SOL"],
    exchange="binance",
    interval="1h",
    start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
    end_time=datetime(2024, 12, 31, tzinfo=timezone.utc),
)

results = vibetrading.backtest.run(code, data=data)
```

### Supported Intervals

| Interval | String |
|---|---|
| 5 minutes | `"5m"` |
| 15 minutes | `"15m"` |
| 1 hour | `"1h"` |
| 4 hours | `"4h"` |
| 1 day | `"1d"` |

## Strategy Generation

### Using Different Models

```python
from vibetrading.strategy import StrategyGenerator

# Anthropic
gen = StrategyGenerator(model="claude-sonnet-4-20250514")
code = gen.generate("BTC momentum strategy")

# OpenAI
gen = StrategyGenerator(model="gpt-4o")
code = gen.generate("ETH mean reversion")
```

### Custom Prompts

```python
from vibetrading.strategy import build_generation_prompt

# Get the prompt messages for use with your own LLM
messages = build_generation_prompt("BTC grid strategy with 1% spacing")

# Use with any LLM client
response = your_llm_client.chat(messages)
```

## Project Structure

```
vibetrading/
├── __init__.py          # Package root, exports @vibe
├── backtest.py          # Backtest convenience API
├── strategy.py          # Strategy generation/validation
├── tools.py             # Data download utilities
├── indicators.py        # Technical indicators
├── sizing.py            # Position sizing functions
├── models.py            # Data models
├── templates/           # Strategy templates
├── _core/               # Core engine internals
├── _agent/              # AI generation pipeline
├── _metrics/            # Metrics calculator
├── _exchanges/          # Exchange adapters
└── _utils/              # Shared utilities
```

Modules prefixed with `_` are internal. Use the public API modules listed above.
