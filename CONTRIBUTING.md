# Contributing to VibeTrading

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/VibeTradingLabs/vibetrading.git
cd vibetrading
pip install -e ".[dev]"
```

## Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=vibetrading --cov-report=term-missing

# Specific test file
pytest tests/test_sandbox.py -v
```

## Linting & Formatting

We use [ruff](https://docs.astral.sh/ruff/) for both linting and formatting:

```bash
# Check for lint errors
ruff check vibetrading/ tests/

# Auto-fix lint errors
ruff check --fix vibetrading/ tests/

# Format code
ruff format vibetrading/ tests/
```

CI runs both `ruff check` and `ruff format --check` — your PR must pass both.

## Making Changes

1. **Fork the repo** and create a branch from `main`.
2. **Write tests** for any new functionality.
3. **Run the full test suite** and ensure all tests pass.
4. **Lint and format** your code with ruff.
5. **Update documentation** if your change affects the public API.
6. **Open a PR** with a clear description of what changed and why.

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — New feature
- `fix:` — Bug fix
- `docs:` — Documentation only
- `test:` — Adding or updating tests
- `refactor:` — Code change that neither fixes a bug nor adds a feature
- `chore:` — Build process, CI, dependency updates
- `ci:` — CI configuration

## Project Structure

```
vibetrading/
├── __init__.py          # Public API: vibe decorator, evolve()
├── strategy.py          # generate(), validate(), analyze()
├── backtest.py          # BacktestEngine, run()
├── evolution.py         # StrategyEvolver, evolve()
├── tools.py             # download_data(), load_csv()
├── cli.py               # Command-line interface
├── templates/           # Built-in strategy templates
├── _agent/              # LLM integration (generator, validator, analyzer, prompts)
├── _core/               # Engine internals (backtest, sandbox, decorator, live runner)
├── _metrics/            # Performance metrics calculator
├── _models/             # Pydantic models (orders, types)
├── _exchanges/          # Exchange adapters (Hyperliquid, Paradex, etc.)
├── _tools/              # Data download and loading utilities
└── _utils/              # Logging, caching, math helpers
```

Modules prefixed with `_` are internal — use the public API (`vibetrading.backtest`, `vibetrading.strategy`, etc.).

## Adding a Strategy Template

1. Create `vibetrading/templates/your_template.py`
2. Include a `TEMPLATE` string with `{parameter}` placeholders
3. Include a `DEFAULTS` dict with default values
4. Include a `generate(**kwargs) -> str` function
5. Register it in `vibetrading/templates/__init__.py`
6. Add tests in `tests/test_templates.py`
7. Ensure the generated code passes `validate_strategy()`

## Adding an Exchange Adapter

1. Create `vibetrading/_exchanges/your_exchange.py`
2. Implement `VibeSandboxBase` (see `_core/sandbox_base.py` for the interface)
3. Register in `vibetrading/_exchanges/__init__.py`
4. Add integration tests

## Reporting Issues

- **Bug reports**: Include steps to reproduce, expected vs actual behavior, and Python version.
- **Feature requests**: Describe the use case and proposed API.
- **Security issues**: Email directly — do not open a public issue.

## Code of Conduct

Be respectful and constructive. We're all here to build something useful.
