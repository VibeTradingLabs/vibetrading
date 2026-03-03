"""Tests for the metrics calculator."""

import math

from vibetrading._metrics.calculator import MetricsCalculator


class _FakeBacktest:
    """Minimal fake to satisfy MetricsCalculator.__init__."""

    def __init__(
        self,
        results=None,
        trades=None,
        funding_payments=None,
        initial_value=10000.0,
        current_value=10000.0,
        interval="1h",
    ):
        self._initial_total_value = initial_value
        self.interval = interval
        self.results = results or []
        self._all_time_peak = None
        self._max_drawdown_observed = None
        self._win_rate_tracker = {
            "winning_trades": 0,
            "losing_trades": 0,
            "last_processed_trade_index": 0,
            "total_closed_trades": 0,
        }
        self._sharpe_tracker = {
            "return_count": 0,
            "mean": 0.0,
            "m2": 0.0,
            "last_total_value": initial_value,
        }
        self._current_value = current_value

        class _FakeSandbox:
            INITIAL_USDC_BALANCE = 10000.0

        _FakeSandbox.trades = trades or []
        _FakeSandbox.funding_payments = funding_payments or []
        _FakeSandbox.total_tx_fees = 0.0
        self.sandbox = _FakeSandbox()

    def _calculate_total_value(self):
        return self._current_value

    def _update_win_rate_tracker_incremental(self):
        pass


class TestMetricsCalculator:
    def test_empty_results_returns_zero_metrics(self):
        bt = _FakeBacktest()
        calc = MetricsCalculator(bt)
        m = calc.calculate()
        assert m["total_return"] == 0.0
        assert m["sharpe_ratio"] == 0.0
        assert m["max_drawdown"] == 0.0
        assert m["sortino_ratio"] == 0.0
        assert m["calmar_ratio"] == 0.0
        assert m["cagr"] == 0.0

    def test_positive_return(self):
        results = [
            {"total_value": 10000.0},
            {"total_value": 10500.0},
            {"total_value": 11000.0},
        ]
        bt = _FakeBacktest(results=results, initial_value=10000.0, current_value=11000.0)
        calc = MetricsCalculator(bt)
        m = calc.calculate()
        assert abs(m["total_return"] - 0.10) < 1e-6

    def test_negative_return(self):
        results = [
            {"total_value": 10000.0},
            {"total_value": 9500.0},
            {"total_value": 9000.0},
        ]
        bt = _FakeBacktest(results=results, initial_value=10000.0, current_value=9000.0)
        calc = MetricsCalculator(bt)
        m = calc.calculate()
        assert abs(m["total_return"] - (-0.10)) < 1e-6

    def test_max_drawdown(self):
        results = [
            {"total_value": 10000.0},
            {"total_value": 12000.0},
            {"total_value": 9000.0},
            {"total_value": 11000.0},
        ]
        bt = _FakeBacktest(results=results, initial_value=10000.0, current_value=11000.0)
        calc = MetricsCalculator(bt)
        m = calc.calculate()
        # Peak was 12000, trough was 9000 => drawdown = (9000-12000)/12000 = -0.25
        assert abs(m["max_drawdown"] - (-0.25)) < 1e-6

    def test_drawdown_duration(self):
        # Peak at step 1, recovery at step 5 => 3 steps in drawdown
        results = [
            {"total_value": 10000.0},
            {"total_value": 12000.0},  # peak
            {"total_value": 11000.0},  # in drawdown
            {"total_value": 10500.0},  # in drawdown
            {"total_value": 12500.0},  # recovered past peak
        ]
        bt = _FakeBacktest(results=results, initial_value=10000.0, current_value=12500.0)
        calc = MetricsCalculator(bt)
        m = calc.calculate()
        # 2 steps in drawdown × 1h interval = 2 hours
        assert m["max_drawdown_duration_hours"] == 2.0

    def test_win_rate_from_trades(self):
        trades = [
            {"action": "sell", "pnl": 100.0, "asset": "BTC"},
            {"action": "sell", "pnl": -50.0, "asset": "BTC"},
            {"action": "sell", "pnl": 200.0, "asset": "BTC"},
        ]
        results = [{"total_value": 10000.0}, {"total_value": 10250.0}]
        bt = _FakeBacktest(results=results, trades=trades, current_value=10250.0)
        calc = MetricsCalculator(bt)
        m = calc.calculate()
        # 2 winning, 1 losing => 2/3 = 0.6667
        assert abs(m["win_rate"] - (2 / 3)) < 0.01

    def test_profit_factor(self):
        trades = [
            {"action": "sell", "pnl": 300.0, "asset": "BTC"},
            {"action": "sell", "pnl": -100.0, "asset": "BTC"},
            {"action": "sell", "pnl": 200.0, "asset": "BTC"},
        ]
        results = [{"total_value": 10000.0}, {"total_value": 10400.0}]
        bt = _FakeBacktest(results=results, trades=trades, current_value=10400.0)
        calc = MetricsCalculator(bt)
        m = calc.calculate()
        # Gross profit 500 / Gross loss 100 = 5.0
        assert abs(m["profit_factor"] - 5.0) < 0.01

    def test_consecutive_streaks(self):
        trades = [
            {"action": "sell", "pnl": 10.0, "asset": "BTC"},
            {"action": "sell", "pnl": 20.0, "asset": "BTC"},
            {"action": "sell", "pnl": 15.0, "asset": "BTC"},
            {"action": "sell", "pnl": -5.0, "asset": "BTC"},
            {"action": "sell", "pnl": -3.0, "asset": "BTC"},
        ]
        results = [{"total_value": 10000.0}, {"total_value": 10037.0}]
        bt = _FakeBacktest(results=results, trades=trades, current_value=10037.0)
        calc = MetricsCalculator(bt)
        m = calc.calculate()
        assert m["max_consecutive_wins"] == 3
        assert m["max_consecutive_losses"] == 2

    def test_funding_revenue(self):
        payments = [
            {"funding_payment": -10.0},  # Received 10
            {"funding_payment": 5.0},  # Paid 5
            {"funding_payment": -20.0},  # Received 20
        ]
        results = [{"total_value": 10000.0}, {"total_value": 10025.0}]
        bt = _FakeBacktest(results=results, funding_payments=payments, current_value=10025.0)
        calc = MetricsCalculator(bt)
        m = calc.calculate()
        # Revenue = -sum of payments = -((-10) + 5 + (-20)) = 25
        assert abs(m["funding_revenue"] - 25.0) < 1e-6

    def test_total_trades_count(self):
        trades = [{"asset": "BTC"}, {"asset": "ETH"}, {"asset": "BTC"}]
        results = [{"total_value": 10000.0}, {"total_value": 10000.0}]
        bt = _FakeBacktest(results=results, trades=trades)
        calc = MetricsCalculator(bt)
        m = calc.calculate()
        assert m["total_trades"] == 3

    def test_expectancy(self):
        trades = [
            {"action": "sell", "pnl": 200.0, "asset": "BTC"},
            {"action": "sell", "pnl": -100.0, "asset": "BTC"},
            {"action": "sell", "pnl": 200.0, "asset": "BTC"},
            {"action": "sell", "pnl": -100.0, "asset": "BTC"},
        ]
        results = [{"total_value": 10000.0}, {"total_value": 10200.0}]
        bt = _FakeBacktest(results=results, trades=trades, current_value=10200.0)
        calc = MetricsCalculator(bt)
        m = calc.calculate()
        # win_rate=0.5, avg_win=200, avg_loss=-100
        # expectancy = 200*0.5 + (-100)*0.5 = 50
        assert abs(m["expectancy"] - 50.0) < 1e-6

    def test_cagr_positive(self):
        # 100 hourly results, 10% total return
        results = [{"total_value": 10000.0 + i * 10} for i in range(100)]
        bt = _FakeBacktest(results=results, initial_value=10000.0, current_value=10990.0)
        calc = MetricsCalculator(bt)
        m = calc.calculate()
        assert m["cagr"] > 0

    def test_sortino_ratio_computed(self):
        # Generate results with some volatility
        import random

        random.seed(42)
        values = [10000.0]
        for _ in range(99):
            values.append(values[-1] * (1 + random.gauss(0.001, 0.01)))
        results = [{"total_value": v} for v in values]
        bt = _FakeBacktest(results=results, initial_value=values[0], current_value=values[-1])
        calc = MetricsCalculator(bt)
        m = calc.calculate()
        # Sortino should be a finite number
        assert math.isfinite(m["sortino_ratio"])

    def test_calmar_ratio_computed(self):
        results = [
            {"total_value": 10000.0},
            {"total_value": 12000.0},
            {"total_value": 10000.0},
            {"total_value": 13000.0},
        ]
        bt = _FakeBacktest(results=results, initial_value=10000.0, current_value=13000.0)
        calc = MetricsCalculator(bt)
        m = calc.calculate()
        assert m["calmar_ratio"] > 0

    def test_largest_win_and_loss(self):
        trades = [
            {"action": "sell", "pnl": 50.0, "asset": "BTC"},
            {"action": "sell", "pnl": 500.0, "asset": "BTC"},
            {"action": "sell", "pnl": -200.0, "asset": "BTC"},
            {"action": "sell", "pnl": -10.0, "asset": "BTC"},
        ]
        results = [{"total_value": 10000.0}, {"total_value": 10340.0}]
        bt = _FakeBacktest(results=results, trades=trades, current_value=10340.0)
        calc = MetricsCalculator(bt)
        m = calc.calculate()
        assert m["largest_win"] == 500.0
        assert m["largest_loss"] == -200.0
