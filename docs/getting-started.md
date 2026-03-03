# Getting Started

Get from zero to running backtest in under 5 minutes.

## Install

```bash
pip install vibetrading
```

## Quick Start: Run a Backtest

```python
import vibetrading.backtest

strategy = """
import math
from vibetrading import vibe, get_perp_price, get_perp_position, get_perp_summary
from vibetrading import set_leverage, long, reduce_position
from vibetrading.indicators import rsi

@vibe(interval="1h")
def my_strategy():
    price = get_perp_price("BTC")
    if math.isnan(price):
        return

    ohlcv = get_futures_ohlcv("BTC", "1h", 20)
    if ohlcv is None or len(ohlcv) < 15:
        return

    current_rsi = rsi(ohlcv["close"], 14).iloc[-1]

    position = get_perp_position("BTC")
    if position and position.get("size", 0) != 0:
        pnl_pct = (price - position["entry_price"]) / position["entry_price"]
        if pnl_pct >= 0.03 or pnl_pct <= -0.02:
            reduce_position("BTC", abs(position["size"]))
        return

    if current_rsi < 30:
        summary = get_perp_summary()
        margin = summary.get("available_margin", 0)
        if margin > 100:
            set_leverage("BTC", 3)
            qty = (margin * 0.1 * 3) / price
            if qty * price >= 15:
                long("BTC", qty, price, order_type="market")
"""

results = vibetrading.backtest.run(
    strategy,
    interval="1h",
    initial_balances={"USDC": 10000},
    slippage_bps=5,
)

if results:
    m = results["metrics"]
    print(f"Return: {m['total_return']:+.2%}")
    print(f"Sharpe: {m['sharpe_ratio']:.3f}")
    print(f"Max DD: {m['max_drawdown']:.2%}")
    print(f"Trades: {m['number_of_trades']}")
```

## Quick Start: Generate with AI

```python
import vibetrading.strategy

# Generate a strategy from natural language
code = vibetrading.strategy.generate(
    "BTC momentum strategy: go long when RSI crosses above 30 from oversold, "
    "use 3x leverage, take profit at 3%, stop loss at 2%"
)

# Validate the generated code
result = vibetrading.strategy.validate(code)
if result.is_valid:
    print("Strategy is valid!")
else:
    print(result.format_for_llm())

# Backtest it
results = vibetrading.backtest.run(code, slippage_bps=5)
```

## Using the CLI

```bash
# Backtest a strategy file
vibetrading backtest my_strategy.py -i 1h

# Validate strategy code
vibetrading validate my_strategy.py

# Download market data
vibetrading download BTC ETH SOL -i 1h -o data/

# Generate from a template
vibetrading template momentum -o my_strategy.py

# Check version
vibetrading version
```

## Using Built-in Indicators

```python
from vibetrading.indicators import sma, ema, rsi, bbands, atr, macd, stochastic, vwap

# All indicators take pandas Series and return pandas Series
ohlcv = get_futures_ohlcv("BTC", "1h", 50)

# Simple/Exponential Moving Average
sma_20 = sma(ohlcv["close"], 20)
ema_12 = ema(ohlcv["close"], 12)

# RSI
rsi_14 = rsi(ohlcv["close"], 14)
current_rsi = rsi_14.iloc[-1]

# Bollinger Bands
upper, middle, lower = bbands(ohlcv["close"], 20, 2.0)

# ATR
atr_14 = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14)

# MACD
macd_line, signal, histogram = macd(ohlcv["close"], 12, 26, 9)

# Stochastic
k, d = stochastic(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14, 3)

# VWAP
vwap_line = vwap(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"])
```

## Position Sizing

```python
from vibetrading.sizing import (
    kelly_size,
    fixed_fraction_size,
    volatility_adjusted_size,
    risk_per_trade_size,
    max_position_size,
)

# Kelly criterion (half-Kelly by default for safety)
size = kelly_size(win_rate=0.55, avg_win=100, avg_loss=80, balance=10000)

# Fixed fraction
size = fixed_fraction_size(balance=10000, fraction=0.02)

# Volatility-adjusted
size = volatility_adjusted_size(balance=10000, atr=500, risk_pct=0.02, price=50000)

# Risk per trade
size = risk_per_trade_size(balance=10000, risk_pct=0.01, stop_distance=1000, price=50000)
```

## Strategy Templates

```python
from vibetrading.templates import momentum, mean_reversion, grid, dca, multi_momentum

# Generate template code
code = momentum()      # Momentum/trend following
code = mean_reversion() # RSI-based mean reversion
code = grid()           # Grid trading
code = dca()            # Dollar cost averaging
code = multi_momentum() # Multi-asset momentum
```

## Next Steps

- Browse the [sample strategies](../strategies/) for production-ready examples
- Read the [API Reference](./api-reference.md) for all available functions
- Check the [Configuration](./configuration.md) guide for advanced settings
- See [CONTRIBUTING.md](../CONTRIBUTING.md) to contribute
