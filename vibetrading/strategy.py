"""
Strategy generation, validation, and prompt templates.

Usage::

    import vibetrading.strategy

    # Generate strategy from natural language
    gen = vibetrading.strategy.StrategyGenerator(model="gpt-4o")
    code = gen.generate("BTC momentum with RSI oversold entry, 3x leverage")

    # Or use the convenience function
    code = vibetrading.strategy.generate("BTC momentum with RSI")

    # Validate strategy code
    result = vibetrading.strategy.validate(code)
    if not result.is_valid:
        print(result.format_for_llm())

    # Use prompt templates with your own LLM client
    messages = vibetrading.strategy.build_generation_prompt("ETH mean reversion")
"""

from ._agent.generator import StrategyGenerator
from ._agent.generator import generate_strategy as generate
from ._agent.validator import validate_strategy as validate
from ._agent.validator import StrategyValidationResult
from ._agent.prompt import (
    STRATEGY_SYSTEM_PROMPT,
    VIBETRADING_API_REFERENCE,
    STRATEGY_CONSTRAINTS,
    build_generation_prompt,
)

__all__ = [
    "StrategyGenerator",
    "generate",
    "validate",
    "StrategyValidationResult",
    "STRATEGY_SYSTEM_PROMPT",
    "VIBETRADING_API_REFERENCE",
    "STRATEGY_CONSTRAINTS",
    "build_generation_prompt",
]
