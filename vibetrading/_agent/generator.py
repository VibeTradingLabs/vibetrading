"""
StrategyGenerator - Generate trading strategies from natural language prompts.

Uses any OpenAI-compatible LLM API (via litellm) to convert user descriptions
into executable vibetrading strategy code. Supports closed-loop validation.

Usage::

    from vibetrading.agent import StrategyGenerator

    generator = StrategyGenerator(model="gpt-4o")
    code = generator.generate("BTC momentum with RSI and ATR stop loss")

    # Or with validation loop
    code = generator.generate("BTC grid strategy", validate=True, max_retries=3)
"""

import logging
from typing import Any

from .prompt import STRATEGY_SYSTEM_PROMPT, build_generation_prompt
from .validator import validate_strategy, StrategyValidationResult

logger = logging.getLogger(__name__)


class StrategyGenerator:
    """
    Generate vibetrading strategies from natural language prompts.

    Wraps any OpenAI-compatible LLM API via ``litellm`` to generate
    strategy code and validate it.

    Args:
        model: LLM model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514",
               "gemini/gemini-2.5-pro"). Passed directly to litellm.
        api_key: API key for the model provider. Can also be set via environment
                 variables (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.).
        temperature: Sampling temperature (default 0.2 for deterministic output).
        **kwargs: Additional keyword arguments passed to litellm.completion().

    Example::

        generator = StrategyGenerator(model="gpt-4o", api_key="sk-...")
        code = generator.generate("BTC momentum with RSI oversold/overbought")
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        temperature: float = 0.2,
        **kwargs,
    ):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.extra_kwargs = kwargs

    def generate(
        self,
        prompt: str,
        *,
        assets: list[str] | None = None,
        market_type: str | None = None,
        max_leverage: int | None = None,
        interval: str | None = None,
        additional_context: str | None = None,
        validate: bool = True,
        max_retries: int = 2,
    ) -> str:
        """
        Generate strategy code from a natural language prompt.

        Args:
            prompt: Natural language description of the strategy.
            assets: Restrict to specific assets (e.g., ["BTC", "ETH"]).
            market_type: "perp" or "spot".
            max_leverage: Maximum leverage allowed.
            interval: Strategy execution interval (e.g., "1h", "5m").
            additional_context: Extra context to include in the prompt.
            validate: Whether to validate the generated code and retry on errors.
            max_retries: Maximum number of retry attempts if validation fails.

        Returns:
            Generated Python strategy code as a string.

        Raises:
            ImportError: If litellm is not installed.
            ValueError: If generation fails after all retries.
        """
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "litellm is required for strategy generation. "
                "Install it with: pip install 'vibetrading[agent]'"
            )

        messages = build_generation_prompt(
            prompt,
            assets=assets,
            market_type=market_type,
            max_leverage=max_leverage,
            interval=interval,
            additional_context=additional_context,
        )

        code = None
        last_validation: StrategyValidationResult | None = None

        for attempt in range(max_retries + 1):
            if attempt > 0 and last_validation and not last_validation.is_valid:
                feedback = last_validation.format_for_llm()
                messages.append({"role": "assistant", "content": code or ""})
                messages.append({"role": "user", "content": feedback})
                logger.info(
                    "Retry %d/%d: feeding validation errors back to LLM",
                    attempt, max_retries,
                )

            completion_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            }
            if self.api_key:
                completion_kwargs["api_key"] = self.api_key
            completion_kwargs.update(self.extra_kwargs)

            response = litellm.completion(**completion_kwargs)
            code = response.choices[0].message.content.strip()
            code = self._clean_code(code)

            if not validate:
                return code

            last_validation = validate_strategy(code)
            if last_validation.is_valid:
                if last_validation.warnings:
                    for w in last_validation.warnings:
                        logger.warning("Strategy validation warning: %s", w)
                return code

            logger.warning(
                "Validation failed (attempt %d/%d): %s",
                attempt + 1, max_retries + 1,
                "; ".join(last_validation.errors),
            )

        if last_validation and not last_validation.is_valid:
            error_summary = "; ".join(last_validation.errors)
            raise ValueError(
                f"Strategy generation failed validation after {max_retries + 1} attempts. "
                f"Errors: {error_summary}"
            )

        return code or ""

    @staticmethod
    def _clean_code(code: str) -> str:
        """Remove markdown fences and XML tags from LLM output."""
        code = code.strip()

        # Remove XML tags (e.g., <update_agent_kernel_code>)
        import re
        code = re.sub(r"</?update_agent_kernel_code>", "", code)

        # Remove markdown code fences
        if code.startswith("```"):
            lines = code.split("\n")
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)

        return code.strip()


def generate_strategy(
    prompt: str,
    *,
    model: str = "gpt-4o",
    api_key: str | None = None,
    validate: bool = True,
    max_retries: int = 2,
    **kwargs,
) -> str:
    """
    Convenience function to generate strategy code from a prompt.

    This is a shortcut that creates a StrategyGenerator and calls generate().

    Args:
        prompt: Natural language description of the strategy.
        model: LLM model identifier (default: "gpt-4o").
        api_key: API key for the model provider.
        validate: Whether to validate and retry on errors.
        max_retries: Maximum retry attempts.
        **kwargs: Passed to StrategyGenerator.generate() (assets, market_type, etc.).

    Returns:
        Generated Python strategy code string.

    Example::

        from vibetrading import generate_strategy

        code = generate_strategy(
            "BTC momentum strategy with RSI oversold entry and ATR stop loss",
            model="gpt-4o",
        )
    """
    generator = StrategyGenerator(model=model, api_key=api_key)
    return generator.generate(prompt, validate=validate, max_retries=max_retries, **kwargs)
