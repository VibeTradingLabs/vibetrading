# API Reference

Complete reference for all public vibetrading APIs.

## Core Modules

### `vibetrading.backtest`

#### `run(strategy_code, *, interval, initial_balances, ...)`

Run a backtest in one call.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `strategy_code` | `str` | required | Python code with `@vibe` decorated function |
| `interval` | `str` | `"1h"` | Candle interval: `"5m"`, `"15m"`, `"1h"`, `"4h"`, `"1d"` |
| `initial_balances` | `dict[str, float]` | `{"USDC": 10000}` | Starting balances |
| `start_time` | `datetime` | `2025-01-01 UTC` | Backtest start time |
| `end_time` | `datetime` | `start + 180 days` | Backtest end time |
| `exchange` | `str` | `"binance"` | Exchange for data download |
| `data` | `dict` | `None` | Pre-loaded data (skips download) |
| `mute_strategy_prints` | `bool` | `False` | Suppress strategy print output |
| `slippage_bps` | `float` | `0.0` | Slippage in basis points for market orders |

**Returns:** `dict` with keys:
- `trades` — list of executed trades
- `metrics` — performance metrics dict (see [Metrics](#metrics))
- `simulation_info` — backtest metadata
- `final_balances` — ending balances by asset
- `equity_curve` — pandas DataFrame with columns: `total_value`, `returns`, `cumulative_returns`, `drawdown`, `peak`

#### `BacktestEngine`

Full-control class for advanced usage.

```python
engine = BacktestEngine(interval="1h", initial_balances={"USDC": 10000})
result = engine.run(strategy_code)
```

---

### `vibetrading.strategy`

#### `generate(prompt, *, model, ...)`

Generate strategy code from natural language.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt` | `str` | required | Natural language strategy description |
| `model` | `str` | `"gpt-4o"` | LLM model to use |

**Returns:** `str` — generated Python strategy code

#### `validate(code)`

Validate strategy code without running it.

**Returns:** `StrategyValidationResult` with:
- `is_valid` — `bool`
- `errors` — list of validation errors
- `warnings` — list of warnings
- `format_for_llm()` — formatted string for LLM feedback

#### `analyze(results, *, strategy_code)`

Analyze backtest results with an LLM.

**Returns:** `BacktestAnalysisResult` with `summary`, `suggestions`, `score`

#### `build_generation_prompt(description)`

Build LLM messages for strategy generation (use with your own LLM client).

**Returns:** `list[dict]` — messages in OpenAI format

---

### `vibetrading.tools`

#### `download_data(assets, *, exchange, interval, start_time, end_time)`

Download OHLCV data from an exchange.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `assets` | `list[str]` | required | Asset symbols: `["BTC", "ETH"]` |
| `exchange` | `str` | `"binance"` | Exchange name |
| `interval` | `str` | `"1h"` | Candle interval |
| `start_time` | `datetime` | 33 days ago | Start time |
| `end_time` | `datetime` | now | End time |

**Returns:** `dict` mapping `"ASSET/interval"` to pandas DataFrames

---

## Strategy Sandbox Functions

These functions are available inside `@vibe`-decorated strategy code:

### Market Data

| Function | Returns | Description |
|---|---|---|
| `get_perp_price(asset)` | `float` | Current perpetual price |
| `get_spot_price(asset)` | `float` | Current spot price |
| `get_price(asset)` | `float` | Current price (spot or perp) |
| `get_futures_ohlcv(asset, interval, limit)` | `DataFrame` | Historical OHLCV candles |
| `get_spot_ohlcv(asset, interval, limit)` | `DataFrame` | Spot OHLCV candles |
| `get_funding_rate(asset)` | `float` | Current funding rate |
| `get_funding_rate_history(asset, limit)` | `list` | Historical funding rates |
| `get_open_interest(asset)` | `float` | Current open interest |
| `get_open_interest_history(asset, limit)` | `list` | Historical open interest |
| `get_current_time()` | `datetime` | Current simulation time |
| `get_supported_assets()` | `list[str]` | Available assets |

### Account

| Function | Returns | Description |
|---|---|---|
| `my_spot_balance(asset?)` | `dict \| float` | Spot balances |
| `my_futures_balance()` | `dict` | Futures account summary |
| `get_perp_summary()` | `dict` | Perp account: `available_margin`, `total_margin`, etc. |
| `get_spot_summary()` | `dict` | Spot account summary |
| `get_perp_position(asset)` | `dict \| None` | Position: `size`, `entry_price`, `pnl`, `leverage` |
| `get_futures_position(asset)` | `float` | Raw position size |

### Trading

| Function | Description |
|---|---|
| `long(asset, qty, price, order_type="market")` | Open/add to long position |
| `short(asset, qty, price, order_type="market")` | Open/add to short position |
| `buy(asset, qty, price, order_type="market")` | Spot buy |
| `sell(asset, qty, price, order_type="market")` | Spot sell |
| `reduce_position(asset, qty)` | Reduce position by quantity |
| `set_leverage(asset, leverage)` | Set leverage (1-100x) |
| `cancel_order(order_id)` | Cancel a specific order |
| `cancel_spot_orders(asset?)` | Cancel all spot orders |
| `cancel_perp_orders(asset?)` | Cancel all perp orders |
| `get_spot_open_orders(asset?)` | List open spot orders |
| `get_perp_open_orders(asset?)` | List open perp orders |

### Order Types

- `"market"` — fills immediately at current price (+ slippage if configured)
- `"limit"` — places at specified price, fills when market reaches it

---

## Indicators

`from vibetrading.indicators import ...`

All indicators take pandas Series and return pandas Series. Pure pandas implementation — no external dependencies.

| Function | Signature | Description |
|---|---|---|
| `sma` | `sma(close, period)` | Simple Moving Average |
| `ema` | `ema(close, period)` | Exponential Moving Average |
| `rsi` | `rsi(close, period=14)` | Relative Strength Index (0-100) |
| `bbands` | `bbands(close, period=20, std=2.0)` | Bollinger Bands → `(upper, middle, lower)` |
| `atr` | `atr(high, low, close, period=14)` | Average True Range |
| `macd` | `macd(close, fast=12, slow=26, signal=9)` | MACD → `(macd_line, signal_line, histogram)` |
| `stochastic` | `stochastic(high, low, close, k=14, d=3)` | Stochastic Oscillator → `(%K, %D)` |
| `vwap` | `vwap(high, low, close, volume)` | Volume Weighted Average Price |

---

## Position Sizing

`from vibetrading.sizing import ...`

| Function | Description |
|---|---|
| `kelly_size(win_rate, avg_win, avg_loss, balance, fraction=0.5)` | Kelly criterion (half-Kelly default) |
| `fixed_fraction_size(balance, fraction)` | Fixed percentage of balance |
| `volatility_adjusted_size(balance, atr, risk_pct, price)` | ATR-normalized sizing |
| `risk_per_trade_size(balance, risk_pct, stop_distance, price)` | Risk per trade |
| `max_position_size(balance, max_pct, price)` | Maximum position cap |

---

## Metrics

Returned by `results["metrics"]` from `backtest.run()`:

| Key | Type | Description |
|---|---|---|
| `total_return` | `float` | Total return as decimal (0.10 = 10%) |
| `cagr` | `float` | Compound Annual Growth Rate |
| `sharpe_ratio` | `float` | Annualized Sharpe ratio |
| `sortino_ratio` | `float` | Annualized Sortino ratio |
| `calmar_ratio` | `float` | Calmar ratio (CAGR / max drawdown) |
| `max_drawdown` | `float` | Maximum drawdown as negative decimal |
| `max_drawdown_duration_hours` | `float` | Longest drawdown in hours |
| `win_rate` | `float` | Win rate (0-1) |
| `profit_factor` | `float` | Gross profit / gross loss |
| `expectancy` | `float` | Expected value per trade in USD |
| `number_of_trades` | `int` | Total trade count |
| `winning_trades` | `int` | Number of winning trades |
| `losing_trades` | `int` | Number of losing trades |
| `avg_win` | `float` | Average winning trade in USD |
| `avg_loss` | `float` | Average losing trade in USD (negative) |
| `largest_win` | `float` | Largest single win |
| `largest_loss` | `float` | Largest single loss (negative) |
| `max_consecutive_wins` | `int` | Longest winning streak |
| `max_consecutive_losses` | `int` | Longest losing streak |
| `total_tx_fees` | `float` | Total transaction fees paid |
| `funding_revenue` | `float` | Net funding revenue |
| `initial_balance` | `float` | Starting balance |
| `total_value` | `float` | Final portfolio value |
| `total_trades` | `int` | Alias for number_of_trades |

---

## Templates

`from vibetrading.templates import ...`

| Function | Description |
|---|---|
| `momentum()` | Trend following with EMA crossover |
| `mean_reversion()` | RSI-based mean reversion |
| `grid()` | Grid trading with price levels |
| `dca()` | Dollar cost averaging |
| `multi_momentum()` | Multi-asset relative strength |

All return valid Python strategy code as `str`.

---

## Supported Exchanges

| Exchange | Spot | Perps | Data Download |
|---|---|---|---|
| Binance | — | — | ✅ |
| Hyperliquid | ✅ | ✅ | ✅ |
| Paradex | — | ✅ | ✅ |
| Lighter | — | ✅ | ✅ |
| Aster | — | ✅ | ✅ |

Backtesting uses Binance data by default. Live trading supports all listed exchanges.
