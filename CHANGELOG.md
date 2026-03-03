# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed
- **`vibetrading.evolution` module**: Removed `StrategyEvolver`, `evolve()`, and related dataclasses (`EvolutionStep`, `EvolutionResult`)
- Removed `vibetrading.evolve()` top-level shortcut
- Removed `examples/07_evolve_strategy.py`

## [0.2.1] - 2026-03-03

### Added
- **Built-in technical indicators** (`vibetrading.indicators`): SMA, EMA, RSI, Bollinger Bands, ATR, MACD, Stochastic, VWAP — pure-pandas, no `ta` library needed
- **Position sizing utilities** (`vibetrading.sizing`): Kelly criterion, fixed fraction, volatility-adjusted (ATR), risk-per-trade, max position size
- **Multi-asset strategy template** (`multi_momentum`): Trade BTC/ETH/SOL simultaneously with independent TP/SL
- **Enhanced analyzer**: Updated scoring guidelines and metrics summary with Sortino, Calmar, profit factor, expectancy, streaks
- **LLM prompt improvements**: Indicators and sizing utilities documented in the generation system prompt
- 63 new tests, 201 total all passing

### Changed
- Updated README with indicators, sizing, and module documentation

## [0.2.0] - 2026-03-03

### Added
- **Enhanced metrics**: Sortino ratio, Calmar ratio, CAGR, profit factor, expectancy, consecutive win/loss streaks, largest win/loss, max drawdown duration, winning/losing trade counts
- **CLI tool**: `vibetrading backtest`, `vibetrading validate`, `vibetrading download`, `vibetrading template`, `vibetrading version` — full terminal workflow with `--json` output
- **Strategy templates**: Built-in momentum, mean reversion, grid, and DCA templates — fully parameterizable, no LLM needed
- **Slippage modeling**: `slippage_bps` parameter for realistic market order simulation
- **Equity curve export**: `results['equity_curve']` DataFrame with total_value, returns, cumulative_returns, drawdown, peak
- **py.typed marker** for PEP 561 typing support
- **CONTRIBUTING.md** with dev setup, project structure, and guidelines
- Test suite: 138 tests covering all core modules
- CI pipeline with GitHub Actions (lint + test on Python 3.10/3.11/3.12)
- Ruff configuration for linting and formatting
- New examples: template quickstart (08) and CLI workflow (09)

### Fixed
- Renamed `examples/07_envole_strategy.py` → `examples/07_evolve_strategy.py` (typo)
- Fixed 464 lint errors across the codebase
- Formatted entire codebase with ruff

## [0.1.6] - 2025-02-26

### Added
- Multi-exchange support: Hyperliquid, Paradex, Lighter, Aster, Extended
- Strategy evolution via iterative generate-backtest-analyze loops
- Backtest engine with realistic simulation (leverage, funding rates, liquidation)
- LLM-powered strategy generation and analysis
- Static validation with closed-loop error feedback
- OHLCV data download and caching via CCXT
- Comprehensive prompt templates and API reference for LLM generation
- Spot and futures trading simulation
- Pydantic models for orders, positions, and account summaries
- `@vibe` decorator for strategy registration
- LiveRunner for real-time strategy execution
