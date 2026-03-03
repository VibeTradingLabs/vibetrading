"""Tests for position sizing utilities."""

from vibetrading.sizing import (
    fixed_fraction_size,
    kelly_size,
    max_position_size,
    risk_per_trade_size,
    volatility_adjusted_size,
)


class TestKellySize:
    def test_positive_edge(self):
        # 55% win rate, avg_win=200, avg_loss=100 => strong edge
        qty = kelly_size(
            win_rate=0.55,
            avg_win=200,
            avg_loss=100,
            balance=10000,
            price=50000,
        )
        assert qty > 0

    def test_no_edge_returns_zero(self):
        # 30% win rate, equal win/loss => negative edge
        qty = kelly_size(
            win_rate=0.30,
            avg_win=100,
            avg_loss=100,
            balance=10000,
            price=50000,
        )
        assert qty == 0.0

    def test_half_kelly_smaller_than_full(self):
        # Use moderate edge so max_fraction doesn't clip
        kwargs = dict(
            win_rate=0.55,
            avg_win=150,
            avg_loss=100,
            balance=10000,
            price=50000,
            max_fraction=0.50,
        )
        half = kelly_size(**kwargs, half_kelly=True)
        full = kelly_size(**kwargs, half_kelly=False)
        assert half < full
        assert abs(half - full / 2) < 1e-10

    def test_respects_max_fraction(self):
        qty = kelly_size(
            win_rate=0.90,
            avg_win=1000,
            avg_loss=10,
            balance=10000,
            price=50000,
            max_fraction=0.05,
            half_kelly=False,
        )
        # Position value should not exceed 5% of balance
        position_value = qty * 50000
        assert position_value <= 10000 * 0.05 + 0.01

    def test_leverage_increases_size(self):
        kwargs = dict(win_rate=0.55, avg_win=200, avg_loss=100, balance=10000, price=50000)
        no_lev = kelly_size(**kwargs, leverage=1)
        with_lev = kelly_size(**kwargs, leverage=3)
        assert with_lev > no_lev

    def test_zero_balance_returns_zero(self):
        assert kelly_size(0.6, 200, 100, 0, 50000) == 0.0

    def test_zero_price_returns_zero(self):
        assert kelly_size(0.6, 200, 100, 10000, 0) == 0.0

    def test_zero_avg_loss_returns_zero(self):
        assert kelly_size(0.6, 200, 0, 10000, 50000) == 0.0


class TestFixedFractionSize:
    def test_basic_calculation(self):
        # 2% of 10000 at price 50000 = 200/50000 = 0.004
        qty = fixed_fraction_size(0.02, 10000, 50000)
        assert abs(qty - 0.004) < 1e-10

    def test_with_leverage(self):
        qty_no_lev = fixed_fraction_size(0.02, 10000, 50000, leverage=1)
        qty_lev = fixed_fraction_size(0.02, 10000, 50000, leverage=3)
        assert abs(qty_lev - qty_no_lev * 3) < 1e-10

    def test_zero_inputs(self):
        assert fixed_fraction_size(0, 10000, 50000) == 0.0
        assert fixed_fraction_size(0.02, 0, 50000) == 0.0
        assert fixed_fraction_size(0.02, 10000, 0) == 0.0


class TestVolatilityAdjustedSize:
    def test_basic_calculation(self):
        # ATR=500, balance=10000, risk=2%, ATR mult=2 => stop=1000
        # risk_amount = 200, qty = 200/1000 = 0.2
        qty = volatility_adjusted_size(
            atr=500,
            balance=10000,
            price=50000,
            risk_pct=0.02,
        )
        assert abs(qty - 0.2) < 1e-10

    def test_higher_volatility_smaller_position(self):
        low_vol = volatility_adjusted_size(atr=200, balance=10000, price=50000)
        high_vol = volatility_adjusted_size(atr=800, balance=10000, price=50000)
        assert low_vol > high_vol

    def test_zero_atr_returns_zero(self):
        assert volatility_adjusted_size(0, 10000, 50000) == 0.0

    def test_leverage_increases_size(self):
        no_lev = volatility_adjusted_size(500, 10000, 50000, leverage=1)
        with_lev = volatility_adjusted_size(500, 10000, 50000, leverage=3)
        assert abs(with_lev - no_lev * 3) < 1e-10


class TestRiskPerTradeSize:
    def test_basic_long(self):
        # Balance=10000, risk=1%, entry=50000, stop=49000 => stop_dist=1000
        # risk_amount = 100, qty = 100/1000 = 0.1
        qty = risk_per_trade_size(10000, 0.01, 50000, 49000)
        assert abs(qty - 0.1) < 1e-10

    def test_basic_short(self):
        # Entry=50000, stop=51000 => stop_dist=1000 (absolute)
        qty = risk_per_trade_size(10000, 0.01, 50000, 51000)
        assert abs(qty - 0.1) < 1e-10

    def test_tighter_stop_bigger_position(self):
        wide = risk_per_trade_size(10000, 0.01, 50000, 48000)  # 2000 stop
        tight = risk_per_trade_size(10000, 0.01, 50000, 49500)  # 500 stop
        assert tight > wide

    def test_zero_stop_distance_returns_zero(self):
        assert risk_per_trade_size(10000, 0.01, 50000, 50000) == 0.0

    def test_with_leverage(self):
        no_lev = risk_per_trade_size(10000, 0.01, 50000, 49000, leverage=1)
        with_lev = risk_per_trade_size(10000, 0.01, 50000, 49000, leverage=3)
        assert abs(with_lev - no_lev * 3) < 1e-10


class TestMaxPositionSize:
    def test_basic(self):
        # 10000 balance, price 50000, no leverage, 95% max
        qty = max_position_size(10000, 50000)
        expected = (10000 * 0.95) / 50000
        assert abs(qty - expected) < 1e-10

    def test_with_leverage(self):
        qty = max_position_size(10000, 50000, leverage=5)
        expected = (10000 * 5 * 0.95) / 50000
        assert abs(qty - expected) < 1e-10

    def test_custom_max_exposure(self):
        qty = max_position_size(10000, 50000, max_exposure_pct=0.50)
        expected = (10000 * 0.50) / 50000
        assert abs(qty - expected) < 1e-10

    def test_zero_inputs(self):
        assert max_position_size(0, 50000) == 0.0
        assert max_position_size(10000, 0) == 0.0
