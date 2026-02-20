"""
Example 5: Generate a strategy with an LLM and backtest it end-to-end.

Demonstrates the full workflow: generate() -> validate_strategy() -> BacktestEngine.run().
Uses litellm under the hood, so any OpenAI-compatible provider works
(OpenAI, Anthropic, Google, local models, etc.).

Usage:
    # Set an API key for your preferred provider
    export OPENAI_API_KEY=sk-...
    # or
    export ANTHROPIC_API_KEY=sk-ant-...

    python examples/05_generate_and_backtest.py
"""

import os
import sys
from datetime import datetime, timezone

from vibetrading import BacktestEngine, validate_strategy
from vibetrading.agent import StrategyGenerator
from vibetrading.tools import download_data


ASSETS = ["BTC"]
EXCHANGE = "binance"
INTERVAL = "1h"
START = datetime(2025, 1, 1, tzinfo=timezone.utc)
END = datetime(2025, 6, 1, tzinfo=timezone.utc)
INITIAL_BALANCES = {"USDC": 10000}


def detect_model() -> str | None:
    """Return a model identifier based on available API keys."""
    if os.environ.get("OPENAI_API_KEY"):
        return "gpt-4o"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic/claude-sonnet-4-20250514"
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return "gemini/gemini-2.5-pro"
    return None


def print_backtest_results(results: dict) -> None:
    """Pretty-print backtest metrics and simulation info."""
    metrics = results["metrics"]
    print(f"  Total Return:    {metrics['total_return']:.2%}")
    print(f"  Max Drawdown:    {metrics['max_drawdown']:.2%}")
    print(f"  Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
    print(f"  Win Rate:        {metrics['win_rate']:.2%}")
    print(f"  Total Trades:    {metrics['number_of_trades']}")
    print(f"  Funding Revenue: ${metrics['funding_revenue']:.2f}")
    print(f"  Total Fees:      ${metrics['total_tx_fees']:.2f}")
    print(f"  Final Value:     ${metrics['total_value']:.2f}")

    info = results["simulation_info"]
    print(f"\n  Simulation: {info['time_range']}, {info['steps']} steps")
    if info["liquidated"]:
        print(f"  WARNING: Liquidated at {info['liquidation_time']}")


def main():
    print("=" * 60)
    print("Example 5: Generate Strategy + Backtest (End-to-End)")
    print("=" * 60)

    model = detect_model()
    if not model:
        print(
            "\nNo API key found. Set one of these environment variables:\n"
            "  OPENAI_API_KEY\n"
            "  ANTHROPIC_API_KEY\n"
            "  GOOGLE_API_KEY / GEMINI_API_KEY\n"
        )
        sys.exit(1)

    print(f"\nDetected model: {model}")

    prompt = (
        "BTC SMA crossover strategy: go long when SMA(10) crosses above SMA(50), "
        "go short when SMA(10) crosses below SMA(50). "
        "Use 3x leverage, risk 10% of available margin per trade, "
        "take profit at 8%, stop loss at 4%."
    )

    # Step 1: Generate strategy code
    print("\n--- Step 1: Generate strategy ---\n")
    print(f"  Prompt: {prompt}")
    print(f"  Model:  {model}\n")

    generator = StrategyGenerator(model=model, temperature=0.2)
    code = generator.generate(
        prompt=prompt,
    )

    print("  Generated strategy (first 20 lines):")
    for line in code.split("\n"):
        print(f"    {line}")

    # Step 2: Validate
    print("--- Step 2: Validate strategy ---\n")
    validation = validate_strategy(code)
    print(f"  Valid:    {validation.is_valid}")
    print(f"  Errors:   {len(validation.errors)}")
    print(f"  Warnings: {len(validation.warnings)}")
    for e in validation.errors:
        print(f"    ERROR: {e}")
    for w in validation.warnings:
        print(f"    WARN: {w}")

    if not validation.is_valid:
        print("\n  Strategy failed validation, skipping backtest.")
        sys.exit(1)

    # Step 3: Download historical data
    print("\n--- Step 3: Download historical data ---\n")
    data = download_data(
        ASSETS,
        exchange=EXCHANGE,
        start_time=START,
        end_time=END,
        interval=INTERVAL,
        market_type="perp",
    )
    for key, df in data.items():
        print(f"  {key}: {len(df)} rows")

    # Step 4: Backtest
    print("\n--- Step 4: Run backtest ---\n")
    engine = BacktestEngine(
        start_time=START,
        end_time=END,
        interval=INTERVAL,
        exchange=EXCHANGE,
        initial_balances=INITIAL_BALANCES,
        data=data,
    )

    results = engine.run(code)
    if results:
        print_backtest_results(results)
    else:
        print("  Backtest returned no results.")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
