# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Test suite covering validator, decorator, models, sandbox, metrics, backtest engine, and prompt builder
- CI pipeline with GitHub Actions (lint + test on Python 3.10/3.11/3.12)
- Ruff configuration for linting and formatting
- pytest configuration with coverage support
- CHANGELOG.md

### Fixed
- Renamed `examples/07_envole_strategy.py` → `examples/07_evolve_strategy.py` (typo)

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
