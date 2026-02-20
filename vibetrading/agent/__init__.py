"""
Agent module for prompt-to-strategy generation.

Provides the tools for AI agents to generate, validate, and refine
trading strategies from natural language prompts.
"""

from .prompt import (
    STRATEGY_SYSTEM_PROMPT,
    VIBETRADING_API_REFERENCE,
    STRATEGY_CONSTRAINTS,
    build_generation_prompt,
)
from .validator import validate_strategy, StrategyValidationResult
from .generator import StrategyGenerator, generate_strategy

__all__ = [
    # Prompt building
    "STRATEGY_SYSTEM_PROMPT",
    "VIBETRADING_API_REFERENCE",
    "STRATEGY_CONSTRAINTS",
    "build_generation_prompt",
    # Validation
    "validate_strategy",
    "StrategyValidationResult",
    # Generation
    "StrategyGenerator",
    "generate_strategy",
]
