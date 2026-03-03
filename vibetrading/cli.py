"""
vibetrading CLI - Command-line interface for strategy development.

Usage::

    # Download data
    vibetrading download BTC ETH --exchange binance --interval 1h

    # Backtest a strategy file
    vibetrading backtest strategy.py --interval 1h --balance 10000

    # Validate strategy code
    vibetrading validate strategy.py

    # Show version
    vibetrading version
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import vibetrading
import vibetrading.backtest
import vibetrading.strategy
import vibetrading.tools


def _format_pct(value: float) -> str:
    return f"{value:+.2%}"


def _format_usd(value: float) -> str:
    return f"${value:,.2f}"


def _format_ratio(value: float) -> str:
    return f"{value:.4f}"


def _print_metrics_table(metrics: dict) -> None:
    """Print a formatted metrics summary."""
    sections = [
        (
            "Performance",
            [
                ("Total Return", _format_pct(metrics.get("total_return", 0))),
                ("CAGR", _format_pct(metrics.get("cagr", 0))),
                ("Total PnL", _format_usd(metrics.get("total_pnl", 0))),
                ("Initial Balance", _format_usd(metrics.get("initial_balance", 0))),
                ("Final Value", _format_usd(metrics.get("total_value", 0))),
            ],
        ),
        (
            "Risk",
            [
                ("Max Drawdown", _format_pct(metrics.get("max_drawdown", 0))),
                ("Max DD Duration", f"{metrics.get('max_drawdown_duration_hours', 0):.1f}h"),
                ("Sharpe Ratio", _format_ratio(metrics.get("sharpe_ratio", 0))),
                ("Sortino Ratio", _format_ratio(metrics.get("sortino_ratio", 0))),
                ("Calmar Ratio", _format_ratio(metrics.get("calmar_ratio", 0))),
            ],
        ),
        (
            "Trades",
            [
                ("Total Trades", str(metrics.get("number_of_trades", 0))),
                ("Win Rate", _format_pct(metrics.get("win_rate", 0))),
                ("Profit Factor", _format_ratio(metrics.get("profit_factor", 0))),
                ("Expectancy", _format_usd(metrics.get("expectancy", 0))),
                ("Avg Win", _format_usd(metrics.get("avg_win", 0))),
                ("Avg Loss", _format_usd(metrics.get("avg_loss", 0))),
                ("Largest Win", _format_usd(metrics.get("largest_win", 0))),
                ("Largest Loss", _format_usd(metrics.get("largest_loss", 0))),
                ("Max Consec. Wins", str(metrics.get("max_consecutive_wins", 0))),
                ("Max Consec. Losses", str(metrics.get("max_consecutive_losses", 0))),
            ],
        ),
        (
            "Costs",
            [
                ("Total TX Fees", _format_usd(metrics.get("total_tx_fees", 0))),
                ("Funding Revenue", _format_usd(metrics.get("funding_revenue", 0))),
            ],
        ),
    ]

    print("\n" + "=" * 50)
    print("  BACKTEST RESULTS")
    print("=" * 50)

    for section_name, rows in sections:
        print(f"\n  {section_name}")
        print("  " + "-" * 46)
        for label, value in rows:
            print(f"  {label:<24} {value:>22}")

    print("\n" + "=" * 50)


def cmd_backtest(args: argparse.Namespace) -> int:
    """Run a backtest on a strategy file."""
    strategy_path = Path(args.strategy)
    if not strategy_path.exists():
        print(f"Error: Strategy file not found: {strategy_path}", file=sys.stderr)
        return 1

    strategy_code = strategy_path.read_text()

    start_time = None
    end_time = None
    if args.start:
        start_time = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if args.end:
        end_time = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    initial_balances = {"USDC": args.balance}

    # Load pre-downloaded data if specified
    data = None
    if args.data_dir:
        data = _load_data_dir(args.data_dir, args.interval)

    print(f"Running backtest: {strategy_path.name}")
    print(f"  Interval: {args.interval}")
    print(f"  Balance: ${args.balance:,.0f}")
    print(f"  Exchange: {args.exchange}")
    if start_time:
        print(f"  Start: {start_time.date()}")
    if end_time:
        print(f"  End: {end_time.date()}")
    print()

    try:
        result = vibetrading.backtest.run(
            strategy_code,
            interval=args.interval,
            initial_balances=initial_balances,
            start_time=start_time,
            end_time=end_time,
            exchange=args.exchange,
            data=data,
            mute_strategy_prints=not args.verbose,
        )
    except Exception as e:
        print(f"Backtest failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    if result is None:
        print("Backtest returned no results.", file=sys.stderr)
        return 1

    metrics = result.get("metrics", {})
    sim = result.get("simulation_info", {})

    if args.json:
        output = {"metrics": metrics, "simulation_info": sim, "total_trades": len(result.get("trades", []))}
        print(json.dumps(output, indent=2, default=str))
    else:
        _print_metrics_table(metrics)
        print(f"\n  Simulation: {sim.get('steps', 0)} steps | {sim.get('time_range', 'N/A')}")
        if sim.get("liquidated"):
            print(f"  ⚠️  LIQUIDATED at {sim.get('liquidation_time', 'unknown')}")
        print()

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a strategy file."""
    strategy_path = Path(args.strategy)
    if not strategy_path.exists():
        print(f"Error: Strategy file not found: {strategy_path}", file=sys.stderr)
        return 1

    code = strategy_path.read_text()
    result = vibetrading.strategy.validate(code)

    if result.is_valid and not result.warnings:
        print(f"✅ {strategy_path.name}: All checks passed.")
        return 0

    if result.is_valid:
        print(f"✅ {strategy_path.name}: Valid (with warnings)")
    else:
        print(f"❌ {strategy_path.name}: Validation failed")

    for error in result.errors:
        print(f"  ERROR: {error}")
    for warning in result.warnings:
        print(f"  WARNING: {warning}")

    return 0 if result.is_valid else 1


def cmd_download(args: argparse.Namespace) -> int:
    """Download historical data."""
    print(f"Downloading data for {', '.join(args.assets)}")
    print(f"  Exchange: {args.exchange}")
    print(f"  Interval: {args.interval}")
    print(f"  Period: {args.start} to {args.end}")
    print()

    start_time = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_time = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    try:
        result = vibetrading.tools.download_data(
            args.assets,
            exchange=args.exchange,
            interval=args.interval,
            start_time=start_time,
            end_time=end_time,
        )
        if result:
            for key, df in result.items():
                print(f"  ✅ {key}: {len(df)} candles")
        else:
            print("  No data returned.")
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return 1

    return 0


def cmd_version(_args: argparse.Namespace) -> int:
    """Print version."""
    print(f"vibetrading {vibetrading.__version__}")
    return 0


def _load_data_dir(data_dir: str, interval: str) -> dict:
    """Load CSV files from a directory into a data dict."""
    data = {}
    data_path = Path(data_dir)
    if not data_path.exists():
        return data

    for csv_file in data_path.glob("*.csv"):
        try:
            df = vibetrading.tools.load_csv(str(csv_file))
            if not df.empty:
                # Infer asset from filename
                stem = csv_file.stem.upper()
                for sep in ["_", "-"]:
                    if sep in stem:
                        stem = stem.split(sep)[0]
                        break
                data[f"{stem}/{interval}"] = df
        except Exception:
            pass
    return data


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="vibetrading",
        description="Agent-first trading framework: generate, backtest, and deploy strategies.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # backtest
    bt_parser = subparsers.add_parser("backtest", help="Run a backtest on a strategy file")
    bt_parser.add_argument("strategy", help="Path to strategy Python file")
    bt_parser.add_argument("-i", "--interval", default="1h", help="Candle interval (default: 1h)")
    bt_parser.add_argument("-b", "--balance", type=float, default=10000, help="Initial USDC balance (default: 10000)")
    bt_parser.add_argument("-e", "--exchange", default="binance", help="Exchange (default: binance)")
    bt_parser.add_argument("-s", "--start", help="Start date (YYYY-MM-DD)")
    bt_parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    bt_parser.add_argument("-d", "--data-dir", help="Directory with pre-downloaded CSV data")
    bt_parser.add_argument("-v", "--verbose", action="store_true", help="Show strategy print output")
    bt_parser.add_argument("--json", action="store_true", help="Output results as JSON")

    # validate
    val_parser = subparsers.add_parser("validate", help="Validate a strategy file")
    val_parser.add_argument("strategy", help="Path to strategy Python file")

    # download
    dl_parser = subparsers.add_parser("download", help="Download historical market data")
    dl_parser.add_argument("assets", nargs="+", help="Assets to download (e.g., BTC ETH)")
    dl_parser.add_argument("-e", "--exchange", default="binance", help="Exchange (default: binance)")
    dl_parser.add_argument("-i", "--interval", default="1h", help="Candle interval (default: 1h)")
    dl_parser.add_argument("-s", "--start", default="2025-01-01", help="Start date (default: 2025-01-01)")
    dl_parser.add_argument("--end", default="2025-07-01", help="End date (default: 2025-07-01)")

    # version
    subparsers.add_parser("version", help="Show version")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "backtest": cmd_backtest,
        "validate": cmd_validate,
        "download": cmd_download,
        "version": cmd_version,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
