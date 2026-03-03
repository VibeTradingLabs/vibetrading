"""Tests for the strategy comparison module."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from vibetrading.compare import print_table, run, to_dataframe


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


STRATEGY_A = """
import math
from vibetrading import vibe, get_perp_price, get_perp_summary, set_leverage, long, get_perp_position, reduce_position

@vibe
def strategy_a():
    price = get_perp_price("BTC")
    if math.isnan(price):
        return
    position = get_perp_position("BTC")
    if position and position.get("size", 0) != 0:
        pnl = (price - position["entry_price"]) / position["entry_price"]
        if pnl >= 0.02 or pnl <= -0.01:
            reduce_position("BTC", abs(position["size"]))
        return
    summary = get_perp_summary()
    margin = summary.get("available_margin", 0)
    if margin > 100:
        set_leverage("BTC", 2)
        qty = (margin * 0.1 * 2) / price
        if qty * price >= 15:
            long("BTC", qty, price, order_type="market")
"""

STRATEGY_B = """
from vibetrading import vibe, get_perp_price
import math

@vibe
def strategy_b():
    price = get_perp_price("BTC")
    # Passive strategy — does nothing
    pass
"""


class TestCompare:
    def test_run_multiple_strategies(self):
        data = {"BTC/1h": _make_ohlcv()}
        results = run(
            {"Active": STRATEGY_A, "Passive": STRATEGY_B},
            interval="1h",
            initial_balances={"USDC": 10000},
            data=data,
            start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 5, tzinfo=timezone.utc),
        )
        assert "Active" in results
        assert "Passive" in results
        assert results["Active"] is not None
        assert results["Passive"] is not None
        assert "metrics" in results["Active"]
        assert "metrics" in results["Passive"]

    def test_run_handles_error(self, capsys):
        bad_code = "not valid python at all @@@@"
        data = {"BTC/1h": _make_ohlcv()}
        results = run(
            {"Bad": bad_code},
            interval="1h",
            data=data,
        )
        assert results["Bad"] is None
        output = capsys.readouterr().out
        assert "ERROR" in output

    def test_print_table(self, capsys):
        mock_results = {
            "Good": {
                "metrics": {
                    "total_return": 0.05,
                    "sharpe_ratio": 1.5,
                    "sortino_ratio": 2.0,
                    "max_drawdown": -0.03,
                    "win_rate": 0.6,
                    "profit_factor": 1.8,
                    "number_of_trades": 50,
                    "total_value": 10500,
                }
            },
            "Bad": None,
        }
        print_table(mock_results)
        output = capsys.readouterr().out
        assert "Good" in output
        assert "ERROR" in output
        assert "Best Sharpe" in output

    def test_to_dataframe(self):
        mock_results = {
            "A": {
                "metrics": {
                    "total_return": 0.1,
                    "sharpe_ratio": 1.0,
                    "max_drawdown": -0.05,
                    "number_of_trades": 10,
                    "total_value": 11000,
                }
            },
            "B": None,
        }
        df = to_dataframe(mock_results)
        assert len(df) == 2
        assert df.loc["A", "total_return"] == 0.1
        assert bool(df.loc["B", "error"]) is True

    def test_to_dataframe_all_valid(self):
        mock_results = {
            "X": {
                "metrics": {
                    "total_return": 0.05,
                    "cagr": 0.1,
                    "sharpe_ratio": 1.2,
                    "sortino_ratio": 1.5,
                    "calmar_ratio": 2.0,
                    "max_drawdown": -0.04,
                    "win_rate": 0.55,
                    "profit_factor": 1.3,
                    "expectancy": 10,
                    "number_of_trades": 100,
                    "winning_trades": 55,
                    "losing_trades": 45,
                    "avg_win": 50,
                    "avg_loss": -40,
                    "largest_win": 200,
                    "largest_loss": -150,
                    "total_tx_fees": 100,
                    "funding_revenue": 50,
                    "total_value": 10500,
                    "max_drawdown_duration_hours": 24,
                }
            },
        }
        df = to_dataframe(mock_results)
        assert df.loc["X", "sharpe_ratio"] == 1.2
        assert df.loc["X", "calmar_ratio"] == 2.0
        assert "strategy" not in df.columns  # It's the index
