"""
Example 6: Analyze backtest results with an LLM.

Demonstrates a single-shot workflow:
    backtest -> vibetrading.strategy.analyze() (LLM feedback)

Uses litellm under the hood, so any OpenAI-compatible provider works.

Usage:
    export OPENAI_API_KEY=sk-...
    # or
    export ANTHROPIC_API_KEY=sk-ant-...

    python examples/06_analyze_backtest.py
"""

import os
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv

import vibetrading.strategy
import vibetrading.backtest
import vibetrading.tools

load_dotenv()

ASSETS = ["BTC"]
EXCHANGE = "binance"
INTERVAL = "1h"
START = datetime(2025, 1, 1, tzinfo=timezone.utc)
END = datetime(2025, 6, 1, tzinfo=timezone.utc)
INITIAL_BALANCES = {"USDC": 10000}

SAMPLE_STRATEGY = """
import math
from vibetrading import (
    vibe,
    get_perp_price,
    get_futures_ohlcv,
    get_perp_summary,
    get_perp_position,
    long,
    short,
    reduce_position,
    set_leverage,
)

ASSET = "BTC"
LEVERAGE = 3
TP_PCT = 0.08
SL_PCT = 0.04
RISK_PER_TRADE_PCT = 0.10

@vibe(interval="1m")
def strategy():
    price = get_perp_price(ASSET)
    if math.isnan(price):
        return

    summary = get_perp_summary()
    margin = summary.get("available_margin", 0.0)
    position = get_perp_position(ASSET)

    if position:
        size = position.get("size", 0.0)
        entry = position.get("entry_price", 0.0)
        if entry > 0 and size != 0:
            if size > 0:
                pnl_pct = (price - entry) / entry
            else:
                pnl_pct = (entry - price) / entry
            if pnl_pct >= TP_PCT:
                reduce_position(ASSET, abs(size) * 0.5)
                return
            elif pnl_pct <= -SL_PCT:
                reduce_position(ASSET, abs(size))
                return
        return

    ohlcv = get_futures_ohlcv(ASSET, "1h", 60)
    if len(ohlcv) < 50:
        return

    sma_fast = ohlcv["close"].rolling(10).mean().iloc[-1]
    sma_slow = ohlcv["close"].rolling(50).mean().iloc[-1]

    set_leverage(ASSET, LEVERAGE)
    qty = (margin * RISK_PER_TRADE_PCT * LEVERAGE) / price
    if qty * price < 15.0:
        return

    if sma_fast > sma_slow:
        long(ASSET, qty, price=price)
    elif sma_fast < sma_slow:
        short(ASSET, qty, price=price)
"""


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
    print("  Example 6: Backtest Analysis with LLM")
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

    # Download data once
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

    print_section("Part 1: Analyze Backtest Results")
    print("  Running backtest on sample strategy...")
    results = vibetrading.backtest.run(
        SAMPLE_STRATEGY,
        interval=INTERVAL,
        initial_balances=INITIAL_BALANCES,
        start_time=START,
        end_time=END,
        data=data,
    )

    if not results:
        print("  Backtest returned no results.")
        print("\n" + "=" * 60)
        print("  Done.")
        print("=" * 60)
        return

    metrics = results["metrics"]
    print(f"  Return: {metrics['total_return']:.2%}")
    print(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
    print(f"  Max DD: {metrics['max_drawdown']:.2%}")
    print(f"  Trades: {metrics['number_of_trades']}")

    print(f"\n  Analyzing with {model}...\n")
    report = vibetrading.strategy.analyze(
        results,
        strategy_code=SAMPLE_STRATEGY,
        model=model,
        detail_level="standard",
    )

    print(f"  Score: {report.score}/10")
    print(f"  Summary: {report.summary}\n")

    if report.strengths:
        print("  Strengths:")
        for s in report.strengths:
            print(f"    + {s}")

    if report.weaknesses:
        print("\n  Weaknesses:")
        for w in report.weaknesses:
            print(f"    - {w}")

    if report.suggestions:
        print("\n  Suggestions:")
        for i, s in enumerate(report.suggestions, 1):
            print(f"    {i}. {s}")

    print(f"\n  Risk: {report.risk_assessment}")

    # Show how format_for_llm() produces generator feedback
    print_section("LLM Feedback Format (for strategy regeneration)")
    feedback = report.format_for_llm()
    for line in feedback.split("\n")[:20]:
        print(f"  {line}")
    print("  ...")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()

