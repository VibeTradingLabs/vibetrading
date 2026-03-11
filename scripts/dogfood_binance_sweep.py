#!/usr/bin/env python3
"""Dogfood runner: download Binance data and backtest bundled templates.

Goal: stay user-centric by continuously running a realistic "new user" flow:
- download spot data
- run a handful of template strategies
- write a ranked report (ROI + basic risk metrics)

Defaults:
- Exchange: binance
- Market: spot
- Asset: BTC
- Interval: 1h
- Lookback: 180d

Output:
- reports/dogfood_binance_spot_<asset>_<interval>_<days>d_<timestamp>.json
- reports/dogfood_binance_spot_<asset>_<interval>_<days>d_<timestamp>.md

Exit code is non-zero if downloads fail or any backtest errors occur.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone

import vibetrading.backtest as bt
from vibetrading._tools.data_downloader import download_data
from vibetrading.templates import dca, grid, mean_reversion, momentum, multi_momentum


@dataclass
class SweepResult:
    name: str
    params: dict
    roi: float
    final_equity: float
    initial_equity: float
    trades: int
    max_drawdown: float | None
    sharpe: float | None


def _equity_from_metrics(metrics: dict) -> float | None:
    # Backtest metrics include total portfolio value (preferred).
    tv = metrics.get("total_value") if isinstance(metrics, dict) else None
    return float(tv) if tv is not None else None


def main() -> str:
    asset = os.environ.get("ASSET", "BTC").upper()
    interval = os.environ.get("INTERVAL", "1h")
    days = int(os.environ.get("DAYS", "180"))

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)

    data = download_data(
        [asset],
        exchange="binance",
        interval=interval,
        market_type="spot",
        start_time=start_time,
        end_time=end_time,
        include_funding=False,
        include_oi=False,
    )

    initial_balances = {"USDC": 10_000.0}

    candidates: list[tuple[str, str, dict]] = []

    # Keep the sweep intentionally small and stable.
    candidates.append(("dca", dca.generate(asset=asset, interval=interval, buy_amount=50, tp_pct=0.10), {"buy_amount": 50, "tp_pct": 0.10}))
    candidates.append(("momentum", momentum.generate(asset=asset, interval=interval, lookback=20, threshold=0.01), {"lookback": 20, "threshold": 0.01}))
    candidates.append(("mean_reversion", mean_reversion.generate(asset=asset, interval=interval, lookback=20, z_entry=1.5, z_exit=0.5), {"lookback": 20, "z_entry": 1.5, "z_exit": 0.5}))
    candidates.append(("grid", grid.generate(asset=asset, interval=interval, grid_levels=6, grid_spacing=0.01), {"grid_levels": 6, "grid_spacing": 0.01}))
    candidates.append(("multi_momentum", multi_momentum.generate(assets=[asset], interval=interval, lookback=20, threshold=0.01), {"assets": [asset], "lookback": 20, "threshold": 0.01}))

    results: list[SweepResult] = []
    errors: list[str] = []

    for name, code, params in candidates:
        try:
            out = bt.run(
                code,
                interval=interval,
                exchange="binance",
                data=data,
                initial_balances=initial_balances,
                start_time=start_time,
                end_time=end_time,
                mute_strategy_prints=True,
                slippage_bps=5.0,
            )
            if not out:
                raise RuntimeError("backtest returned None")

            metrics = out.get("metrics", {}) or {}

            initial_equity = float(initial_balances["USDC"])
            final_equity = _equity_from_metrics(metrics)
            if final_equity is None:
                # fallback: if metrics missing, at least avoid crashing
                final_equity = float(out.get("final_balances", {}).get("USDC", 0.0))

            # Prefer library metric (already accounts for non-USDC holdings)
            roi = metrics.get("total_return")
            if roi is None:
                roi = (final_equity - initial_equity) / initial_equity if initial_equity else 0.0

            trades = int(metrics.get("number_of_trades", 0) or 0)
            max_dd = metrics.get("max_drawdown")
            sharpe = metrics.get("sharpe_ratio")

            results.append(
                SweepResult(
                    name=name,
                    params=params,
                    roi=float(roi),
                    final_equity=float(final_equity),
                    initial_equity=float(initial_equity),
                    trades=trades,
                    max_drawdown=float(max_dd) if max_dd is not None else None,
                    sharpe=float(sharpe) if sharpe is not None else None,
                )
            )
        except Exception as e:
            errors.append(f"{name}: {e}")

    results.sort(key=lambda r: r.roi, reverse=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    reports_dir = os.path.join(os.getcwd(), "reports")
    os.makedirs(reports_dir, exist_ok=True)

    stem = f"dogfood_binance_spot_{asset}_{interval}_{days}d_{ts}"
    json_path = os.path.join(reports_dir, f"{stem}.json")
    md_path = os.path.join(reports_dir, f"{stem}.md")

    payload = {
        "exchange": "binance",
        "market_type": "spot",
        "asset": asset,
        "interval": interval,
        "days": days,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "slippage_bps": 5.0,
        "results": [asdict(r) for r in results],
        "errors": errors,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append(f"# Dogfood sweep — Binance spot {asset} {interval} (last {days}d)\n")
    if errors:
        lines.append("## Errors\n")
        for e in errors:
            lines.append(f"- {e}")
        lines.append("")

    lines.append("## Leaderboard (sorted by ROI)\n")
    for r in results:
        lines.append(
            f"- **{r.name}** roi={r.roi:.2%} equity={r.final_equity:.2f} trades={r.trades} "
            f"maxDD={r.max_drawdown} sharpe={r.sharpe} params={r.params}"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Print paths for CI/cron consumption
    print(json_path)
    print(md_path)

    if errors:
        raise SystemExit(2)

    return md_path


if __name__ == "__main__":
    main()
