"""Tests for built-in technical indicators."""

import math

import numpy as np
import pandas as pd

from vibetrading.indicators import atr, bbands, ema, macd, rsi, sma, stochastic, vwap


def _make_series(n=100, base=50000, seed=42):
    """Generate a synthetic close price series."""
    rng = np.random.default_rng(seed)
    return pd.Series(base + np.cumsum(rng.normal(0, 100, n)))


def _make_ohlcv(n=100, base=50000, seed=42):
    """Generate synthetic OHLCV data."""
    rng = np.random.default_rng(seed)
    close = base + np.cumsum(rng.normal(0, 100, n))
    return pd.DataFrame(
        {
            "open": close - rng.uniform(0, 50, n),
            "high": close + rng.uniform(0, 100, n),
            "low": close - rng.uniform(0, 100, n),
            "close": close,
            "volume": rng.uniform(100, 10000, n),
        }
    )


class TestSMA:
    def test_basic(self):
        s = pd.Series([1, 2, 3, 4, 5], dtype=float)
        result = sma(s, 3)
        assert math.isnan(result.iloc[0])
        assert math.isnan(result.iloc[1])
        assert abs(result.iloc[2] - 2.0) < 1e-10
        assert abs(result.iloc[3] - 3.0) < 1e-10
        assert abs(result.iloc[4] - 4.0) < 1e-10

    def test_length_matches_input(self):
        s = _make_series()
        result = sma(s, 20)
        assert len(result) == len(s)

    def test_nan_for_insufficient_data(self):
        s = _make_series(50)
        result = sma(s, 20)
        assert result.iloc[:19].isna().all()
        assert result.iloc[19:].notna().all()


class TestEMA:
    def test_basic(self):
        s = _make_series()
        result = ema(s, 20)
        assert len(result) == len(s)
        # EMA should have no NaN (unlike SMA)
        assert result.notna().all()

    def test_shorter_period_more_responsive(self):
        s = _make_series()
        ema_fast = ema(s, 5)
        ema_slow = ema(s, 50)
        # Fast EMA should be closer to last price
        last_price = s.iloc[-1]
        assert abs(ema_fast.iloc[-1] - last_price) < abs(ema_slow.iloc[-1] - last_price)


class TestRSI:
    def test_range_0_to_100(self):
        s = _make_series()
        result = rsi(s, 14)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_overbought_on_rising_prices(self):
        # Mostly rising prices with small noise should give high RSI
        rng = np.random.default_rng(42)
        s = pd.Series(np.cumsum(rng.uniform(0.5, 2.0, 100)))
        result = rsi(s, 14)
        assert result.iloc[-1] > 70

    def test_oversold_on_falling_prices(self):
        # Mostly falling prices should give low RSI
        rng = np.random.default_rng(42)
        s = pd.Series(1000 + np.cumsum(rng.uniform(-2.0, -0.5, 100)))
        result = rsi(s, 14)
        assert result.iloc[-1] < 30

    def test_default_period_14(self):
        s = _make_series()
        r1 = rsi(s)
        r2 = rsi(s, 14)
        assert (r1.dropna() == r2.dropna()).all()


class TestBBands:
    def test_upper_above_lower(self):
        s = _make_series()
        upper, middle, lower = bbands(s, 20, 2.0)
        valid_idx = upper.notna() & lower.notna()
        assert (upper[valid_idx] >= lower[valid_idx]).all()

    def test_middle_is_sma(self):
        s = _make_series()
        _, middle, _ = bbands(s, 20, 2.0)
        expected_sma = sma(s, 20)
        valid = middle.notna()
        pd.testing.assert_series_equal(middle[valid], expected_sma[valid])

    def test_wider_std_wider_bands(self):
        s = _make_series()
        u1, _, l1 = bbands(s, 20, 1.0)
        u2, _, l2 = bbands(s, 20, 3.0)
        valid = u1.notna() & u2.notna()
        # Wider std => wider band width
        assert ((u2[valid] - l2[valid]) >= (u1[valid] - l1[valid])).all()


class TestATR:
    def test_positive_values(self):
        df = _make_ohlcv()
        result = atr(df["high"], df["low"], df["close"], 14)
        valid = result.dropna()
        assert (valid > 0).all()

    def test_length_matches(self):
        df = _make_ohlcv()
        result = atr(df["high"], df["low"], df["close"])
        assert len(result) == len(df)


class TestMACD:
    def test_basic_structure(self):
        s = _make_series()
        macd_line, signal_line, histogram = macd(s)
        assert len(macd_line) == len(s)
        assert len(signal_line) == len(s)
        assert len(histogram) == len(s)

    def test_histogram_is_difference(self):
        s = _make_series()
        macd_line, signal_line, histogram = macd(s)
        diff = macd_line - signal_line
        pd.testing.assert_series_equal(histogram, diff)

    def test_default_params(self):
        s = _make_series()
        m1, s1, h1 = macd(s)
        m2, s2, h2 = macd(s, 12, 26, 9)
        pd.testing.assert_series_equal(m1, m2)


class TestStochastic:
    def test_range_0_to_100(self):
        df = _make_ohlcv()
        k, d = stochastic(df["high"], df["low"], df["close"])
        valid_k = k.dropna()
        assert (valid_k >= 0).all()
        assert (valid_k <= 100).all()

    def test_d_is_smoothed_k(self):
        df = _make_ohlcv()
        k, d = stochastic(df["high"], df["low"], df["close"])
        # %D should be smoother (lower variance)
        valid = k.notna() & d.notna()
        assert k[valid].std() >= d[valid].std()


class TestVWAP:
    def test_basic(self):
        df = _make_ohlcv()
        result = vwap(df["high"], df["low"], df["close"], df["volume"])
        assert len(result) == len(df)
        assert result.notna().all()

    def test_within_high_low_range(self):
        df = _make_ohlcv()
        result = vwap(df["high"], df["low"], df["close"], df["volume"])
        # VWAP should generally be within the cumulative high/low range
        # Just check it's reasonable (close to the typical price)
        typical = (df["high"] + df["low"] + df["close"]) / 3
        assert abs(result.iloc[-1] - typical.mean()) < typical.std() * 3
