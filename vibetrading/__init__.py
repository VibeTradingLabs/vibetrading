"""
VibeTrading - Agent-first trading framework for prompt-to-strategy generation.

Describe strategies in natural language. Generate executable Python code.
Backtest and deploy to any supported exchange with the same code.

Quick start::

    from vibetrading import StrategyGenerator, BacktestEngine

    # Generate strategy from prompt
    generator = StrategyGenerator(model="gpt-4o")
    code = generator.generate("BTC momentum with RSI oversold entry, 3x leverage")

    # Backtest
    engine = BacktestEngine(interval="1h", initial_balances={"USDC": 10000})
    results = engine.run(code)
    print(results["metrics"])

    # Or use the prompt template directly with any LLM
    from vibetrading.agent import STRATEGY_SYSTEM_PROMPT, build_generation_prompt
    messages = build_generation_prompt("ETH mean reversion with Bollinger Bands")
"""

__version__ = "0.1.0"

# Core abstractions
from .core.sandbox_base import VibeSandboxBase, SUPPORTED_INTERVALS, SUPPORTED_LEVERAGE
from .core.decorator import vibe

# Backtest engine
from .core.backtest import BacktestEngine
from .core.static_sandbox import StaticSandbox

# Live execution
from .core.live_runner import LiveRunner

# Exchange factory
from .exchanges import create_sandbox, SUPPORTED_EXCHANGES

# Models
from .models.orders import (
    PerpAccountSummary,
    PerpPositionSummary,
    SpotAccountSummary,
    SpotBalanceSummary,
    SpotOrder,
    PerpOrder,
)
from .models.types import SpotMeta, PerpMeta

# Metrics
from .metrics.calculator import MetricsCalculator

# Data
from .tools.data_loader import DEFAULT_PERP_SYMBOLS, DEFAULT_SPOT_SYMBOLS

# Agent - strategy generation from prompts
from .agent.generator import StrategyGenerator, generate_strategy
from .agent.validator import validate_strategy, StrategyValidationResult
from .agent.prompt import (
    STRATEGY_SYSTEM_PROMPT,
    VIBETRADING_API_REFERENCE,
    STRATEGY_CONSTRAINTS,
    build_generation_prompt,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "VibeSandboxBase",
    "SUPPORTED_INTERVALS",
    "SUPPORTED_LEVERAGE",
    "vibe",
    "BacktestEngine",
    "StaticSandbox",
    "LiveRunner",
    # Exchanges
    "create_sandbox",
    "SUPPORTED_EXCHANGES",
    # Models
    "PerpAccountSummary",
    "PerpPositionSummary",
    "SpotAccountSummary",
    "SpotBalanceSummary",
    "SpotOrder",
    "PerpOrder",
    "SpotMeta",
    "PerpMeta",
    # Metrics
    "MetricsCalculator",
    # Data
    "DEFAULT_PERP_SYMBOLS",
    "DEFAULT_SPOT_SYMBOLS",
    # Agent
    "StrategyGenerator",
    "generate_strategy",
    "validate_strategy",
    "StrategyValidationResult",
    "STRATEGY_SYSTEM_PROMPT",
    "VIBETRADING_API_REFERENCE",
    "STRATEGY_CONSTRAINTS",
    "build_generation_prompt",
]
