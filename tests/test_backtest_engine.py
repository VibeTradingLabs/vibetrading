"""Tests for the BacktestEngine end-to-end."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone

from vibetrading._core.backtest import BacktestEngine


def _make_ohlcv(
    start: str = "2025-01-01",
    periods: int = 500,
    freq: str = "1h",
    base_price: float = 50000.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range(start, periods=periods, freq=freq, tz=timezone.utc)
    close = base_price + np.cumsum(rng.normal(0, 100, periods))
    return pd.DataFrame(
        {
            "open": close - rng.uniform(0, 50, periods),
            "high": close + rng.uniform(0, 100, periods),
            "low": close - rng.uniform(0, 100, periods),
            "close": close,
            "volume": rng.uniform(100, 10000, periods),
            "fundingRate": rng.uniform(-0.001, 0.001, periods),
            "openInterest": rng.uniform(1e6, 1e8, periods),
        },
        index=idx,
    )


SIMPLE_STRATEGY = """
import math
from vibetrading import vibe, get_perp_price, get_perp_summary, set_leverage, long, reduce_position, get_perp_position

@vibe
def strategy():
    price = get_perp_price("BTC")
    if math.isnan(price):
        return

    position = get_perp_position("BTC")

    if position:
        entry = position["entry_price"]
        size = position["size"]
        pnl_pct = (price - entry) / entry if entry > 0 else 0
        if pnl_pct >= 0.02:
            reduce_position("BTC", abs(size))
            return
        if pnl_pct <= -0.01:
            reduce_position("BTC", abs(size))
            return
        return

    summary = get_perp_summary()
    margin = summary.get("available_margin", 0)
    if margin < 100:
        return

    set_leverage("BTC", 3)
    qty = (margin * 0.1 * 3) / price
    if qty * price >= 15:
        long("BTC", qty, price, order_type="market")
"""


class TestBacktestEngine:
    def test_engine_init(self):
        data = {"BTC/1h": _make_ohlcv()}
        engine = BacktestEngine(interval="1h", data=data)
        assert engine.interval == "1h"

    def test_unsupported_interval_raises(self):
        with pytest.raises(ValueError, match="Unsupported interval"):
            BacktestEngine(interval="3h")

    def test_run_simple_strategy(self):
        data = {"BTC/1h": _make_ohlcv()}
        engine = BacktestEngine(
            interval="1h",
            initial_balances={"USDC": 10000},
            start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 5, tzinfo=timezone.utc),
            data=data,
        )
        result = engine.run(SIMPLE_STRATEGY)
        assert result is not None
        assert "trades" in result
        assert "metrics" in result
        assert "simulation_info" in result
        assert "final_balances" in result

    def test_run_returns_metrics(self):
        data = {"BTC/1h": _make_ohlcv()}
        engine = BacktestEngine(
            interval="1h",
            initial_balances={"USDC": 10000},
            start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 5, tzinfo=timezone.utc),
            data=data,
        )
        result = engine.run(SIMPLE_STRATEGY)
        metrics = result["metrics"]
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        assert "number_of_trades" in metrics

    def test_no_vibe_decorator_raises(self):
        bad_code = """
import vibetrading

def strategy():
    pass
"""
        data = {"BTC/1h": _make_ohlcv()}
        engine = BacktestEngine(interval="1h", data=data)
        with pytest.raises(ValueError, match="No strategy functions registered"):
            engine.run(bad_code)

    def test_simulation_info_populated(self):
        data = {"BTC/1h": _make_ohlcv()}
        engine = BacktestEngine(
            interval="1h",
            initial_balances={"USDC": 10000},
            start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 3, tzinfo=timezone.utc),
            data=data,
        )
        result = engine.run(SIMPLE_STRATEGY)
        sim = result["simulation_info"]
        assert sim["interval"] == "1h"
        assert sim["liquidated"] is False
        assert sim["steps"] > 0

    def test_mute_strategy_prints(self):
        data = {"BTC/1h": _make_ohlcv()}
        engine = BacktestEngine(
            interval="1h",
            data=data,
            mute_strategy_prints=True,
            start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 2, tzinfo=timezone.utc),
        )
        result = engine.run(SIMPLE_STRATEGY)
        assert result is not None
