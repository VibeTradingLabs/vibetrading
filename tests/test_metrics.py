"""Tests for the metrics calculator."""

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
