# Sample Strategies

Ready-to-run strategies for backtesting and learning. Each file is a complete, self-contained strategy.

## Strategies

| Strategy | Type | Description |
|---|---|---|
| `rsi_mean_reversion.py` | Perp | RSI oversold/overbought entries with Bollinger Band confirmation |
| `macd_trend_follower.py` | Perp | MACD crossover trend following with ATR-based stops |
| `funding_rate_arb.py` | Perp | Captures funding rate payments by positioning against the crowd |
| `multi_asset_momentum.py` | Perp | Rotational momentum across BTC/ETH/SOL with relative strength |
| `breakout_consolidation.py` | Perp | Volatility breakout from consolidation ranges |
| `spot_dca_rebalance.py` | Spot | DCA with periodic rebalancing across a portfolio |

## Running

```bash
# Validate first
vibetrading validate strategies/rsi_mean_reversion.py

# Backtest
vibetrading backtest strategies/rsi_mean_reversion.py --interval 1h --balance 10000

# With custom dates
vibetrading backtest strategies/macd_trend_follower.py -i 1h -b 10000 -s 2025-01-01 --end 2025-06-01
```

## Disclaimer

These are educational examples. Past backtest performance does not guarantee future results. Always understand a strategy before risking real capital.
