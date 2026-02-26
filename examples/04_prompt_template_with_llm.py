"""
Example 4: Use VibeTrading prompt templates with your own LLM client.

Shows how to use vibetrading.strategy.build_generation_prompt() and
vibetrading.strategy.validate() directly, without the built-in
StrategyGenerator. This lets you integrate VibeTrading into any LLM
pipeline (OpenAI, Anthropic, local models, etc.).

Usage:
    # Set your API key first
    export OPENAI_API_KEY=sk-...

    python examples/04_prompt_template_with_llm.py
"""

import os
from dotenv import load_dotenv

import vibetrading.strategy

load_dotenv()

def generate_with_openai(user_prompt: str) -> str:
    """Generate strategy code using OpenAI's API."""
    try:
        import openai
    except ImportError:
        raise ImportError("pip install openai")

    messages = vibetrading.strategy.build_generation_prompt(
        user_prompt,
        assets=["BTC"],
        market_type="perp",
        max_leverage=5,
    )

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def generate_with_anthropic(user_prompt: str) -> str:
    """Generate strategy code using Anthropic's API."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")

    messages = vibetrading.strategy.build_generation_prompt(
        user_prompt,
        assets=["BTC"],
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=messages[0]["content"],
        messages=[{"role": "user", "content": messages[1]["content"]}],
    )
    return response.content[0].text


def closed_loop_generation(user_prompt: str, max_retries: int = 2) -> str:
    """
    Generate with validation feedback loop.

    If validation fails, feed errors back to the LLM and retry.
    Works with any LLM — just replace the call_llm() function.
    """
    try:
        import openai
    except ImportError:
        raise ImportError("pip install openai")

    client = openai.OpenAI()
    messages = vibetrading.strategy.build_generation_prompt(
        user_prompt,
        assets=["BTC"],
        max_leverage=5,
    )

    for attempt in range(max_retries + 1):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.2,
        )
        code = response.choices[0].message.content.strip()

        # Remove markdown fences if present
        if code.startswith("```"):
            lines = code.split("\n")
            lines = lines[1:] if lines[0].startswith("```") else lines
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)

        result = vibetrading.strategy.validate(code)
        if result.is_valid:
            print(f"  Validation passed on attempt {attempt + 1}")
            return code

        print(f"  Attempt {attempt + 1} failed: {'; '.join(result.errors)}")

        # Feed errors back to LLM
        messages.append({"role": "assistant", "content": code})
        messages.append({"role": "user", "content": result.format_for_llm()})

    raise ValueError(f"Generation failed after {max_retries + 1} attempts")


def main():
    print("=" * 60)
    print("Example 4: Prompt Templates with Custom LLM")
    print("=" * 60)

    # Show the prompt structure
    print("\n--- Prompt structure ---\n")
    messages = vibetrading.strategy.build_generation_prompt(
        "BTC momentum with RSI oversold entry",
        assets=["BTC"],
        market_type="perp",
        max_leverage=5,
    )
    print(f"Messages count: {len(messages)}")
    print(f"System prompt length: {len(messages[0]['content'])} chars")
    print(f"User prompt preview: {messages[1]['content'][:200]}...")

    # Check for API key
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))

    if not has_openai and not has_anthropic:
        print("\n--- No API key found ---")
        print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY to run the full example.")
        print("Showing prompt template only.\n")

        print("System prompt (first 500 chars):")
        print(vibetrading.strategy.STRATEGY_SYSTEM_PROMPT[:500])
        print("...\n")
        return

    # Generate with available provider
    user_prompt = "BTC momentum: go long when RSI(14) drops below 30 and SMA(10) > SMA(20), " \
                  "3x leverage, 10% position size, 8% TP, 4% SL"

    if has_openai:
        print("\n--- Generating with OpenAI ---\n")
        code = generate_with_openai(user_prompt)

        print("Generated code (first 30 lines):")
        for i, line in enumerate(code.split("\n")):
            print(f"  {line}")
        print("  ...")

        print("\n--- Validating ---\n")
        result = vibetrading.strategy.validate(code)
        print(f"Valid: {result.is_valid}")
        if result.errors:
            for e in result.errors:
                print(f"  ERROR: {e}")
        if result.warnings:
            for w in result.warnings:
                print(f"  WARN: {w}")

        # Closed-loop demo
        print("\n--- Closed-loop generation ---\n")
        try:
            validated_code = closed_loop_generation(user_prompt)
            print(f"Final code length: {len(validated_code)} chars")
        except ValueError as e:
            print(f"Failed: {e}")

    elif has_anthropic:
        print("\n--- Generating with Anthropic ---\n")
        code = generate_with_anthropic(user_prompt)

        print("Generated code (first 30 lines):")
        for i, line in enumerate(code.split("\n")[:30]):
            print(f"  {line}")
        print("  ...")

        print("\n--- Validating ---\n")
        result = vibetrading.strategy.validate(code)
        print(f"Valid: {result.is_valid}")
        if result.errors:
            for e in result.errors:
                print(f"  ERROR: {e}")
        if result.warnings:
            for w in result.warnings:
                print(f"  WARN: {w}")


if __name__ == "__main__":
    main()
