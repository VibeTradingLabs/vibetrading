"""
Example 7: Evolve a strategy with vibetrading.evolve().

Demonstrates iterative strategy improvement:
    prompt -> generate -> backtest -> analyze -> regenerate (loop)

Uses litellm under the hood, so any OpenAI-compatible provider works.

Usage:
    export OPENAI_API_KEY=sk-...
    # or
    export ANTHROPIC_API_KEY=sk-ant-...

    python examples/07_envole_strategy.py
"""

import os
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv

import vibetrading
import vibetrading.tools

load_dotenv()

ASSETS = ["BTC"]
EXCHANGE = "binance"
INTERVAL = "1h"
START = datetime(2025, 1, 1, tzinfo=timezone.utc)
END = datetime(2025, 6, 1, tzinfo=timezone.utc)
INITIAL_BALANCES = {"USDC": 10000}


def detect_model() -> str | None:
    if os.environ.get("OPENAI_API_KEY"):
        return "gpt-4o"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic/claude-sonnet-4-20250514"
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return "gemini/gemini-2.5-pro"
    if os.environ.get("DEEPSEEK_API_KEY"):
        return "deepseek/deepseek-chat"
    return None


def print_section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}\n")


def main():
    print("=" * 60)
    print("  Example 7: Strategy Evolution with vibetrading.evolve()")
    print("=" * 60)

    model = detect_model()
    if not model:
        print(
            "\nNo API key found. Set one of these environment variables:\n"
            "  OPENAI_API_KEY\n"
            "  ANTHROPIC_API_KEY\n"
            "  GOOGLE_API_KEY / GEMINI_API_KEY\n"
            "  DEEPSEEK_API_KEY"
        )
        sys.exit(1)

    print(f"\n  Detected model: {model}")

    # Download data once for evolution run
    print_section("Downloading historical data")
    data = vibetrading.tools.download_data(
        ASSETS,
        exchange=EXCHANGE,
        start_time=START,
        end_time=END,
        interval=INTERVAL,
        market_type="perp",
    )
    for key, df in data.items():
        print(f"  {key}: {len(df)} rows")

    print_section("Evolving strategy")

    prompt = (
        "BTC momentum strategy: use RSI(14) for overbought/oversold signals "
        "and SMA(20)/SMA(50) crossover for trend confirmation. "
        "3x leverage, 10% risk per trade, TP at 8%, SL at 4%."
    )
    print(f"  Prompt: {prompt}")
    print(f"  Model:  {model}")
    print(f"  Iterations: 3\n")

    def on_step(step):
        status = f"score={step.score}/10" if step.analysis else f"error={step.error}"
        print(f"  [Iteration {step.iteration}] {status}")

    result = vibetrading.evolve(
        prompt,
        iterations=3,
        model=model,
        interval=INTERVAL,
        initial_balances=INITIAL_BALANCES,
        start_time=START,
        end_time=END,
        data=data,
        assets=ASSETS,
        market_type="perp",
        score_threshold=8,
        on_iteration=on_step,
    )

    print(f"\n  Evolution complete:")
    print(f"    Best iteration: {result.best_iteration}")
    print(f"    Best score:     {result.best_score}/10")
    print(f"    Improved:       {result.improved}")
    print(f"    Scores:         {[s.score for s in result.history]}")

    if result.best_metrics:
        m = result.best_metrics
        print(f"\n  Best metrics:")
        print(f"    Return:  {m.get('total_return', 0):.2%}")
        print(f"    Sharpe:  {m.get('sharpe_ratio', 0):.2f}")
        print(f"    Max DD:  {m.get('max_drawdown', 0):.2%}")
        print(f"    Trades:  {m.get('number_of_trades', 0)}")

    # if result.best_code:
    #     print(f"\n  Best strategy code (first 15 lines):")
    #     for line in result.best_code.split("\n")[:15]:
    #         print(f"    {line}")
    #     print("    ...")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()

