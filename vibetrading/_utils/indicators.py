"""
Lightweight technical indicators for strategy development.

These are pure-pandas implementations that don't require the ``ta`` library.
Designed for use inside @vibe strategy functions.

Usage::

    from vibetrading._utils.indicators import rsi, sma, ema, bbands, atr, macd

    ohlcv = get_futures_ohlcv("BTC", "1h", 50)
    rsi_values = rsi(ohlcv["close"], period=14)
    upper, middle, lower = bbands(ohlcv["close"], period=20, std=2.0)
"""

import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average.

    Args:
        series: Price series.
        period: Lookback period.

    Returns:
        SMA series (NaN for first ``period - 1`` values).
    """
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average.

    Args:
        series: Price series.
        period: Lookback period.

    Returns:
        EMA series.
    """
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index.

    Uses the standard Wilder smoothing method (EMA of gains/losses).

    Args:
        series: Price series (typically close prices).
        period: RSI period (default: 14).

    Returns:
        RSI series (0-100 scale, NaN for insufficient data).
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # When avg_loss is 0, RSI is 100 (all gains, no losses)
    rsi_series = pd.Series(index=series.index, dtype=float)
    zero_loss = avg_loss == 0
    nonzero_loss = ~zero_loss & avg_loss.notna()
    rsi_series[zero_loss & avg_gain.notna()] = 100.0
    rs = avg_gain[nonzero_loss] / avg_loss[nonzero_loss]
    rsi_series[nonzero_loss] = 100 - (100 / (1 + rs))
    return rsi_series


def bbands(
    series: pd.Series,
    period: int = 20,
    std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands.

    Args:
        series: Price series.
        period: SMA period (default: 20).
        std: Standard deviation multiplier (default: 2.0).

    Returns:
        Tuple of (upper_band, middle_band, lower_band).
    """
    middle = sma(series, period)
    rolling_std = series.rolling(window=period, min_periods=period).std()
    upper = middle + (std * rolling_std)
    lower = middle - (std * rolling_std)
    return upper, middle, lower


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: ATR period (default: 14).

    Returns:
        ATR series.
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Moving Average Convergence Divergence.

    Args:
        series: Price series.
        fast: Fast EMA period (default: 12).
        slow: Slow EMA period (default: 26).
        signal: Signal line period (default: 9).

    Returns:
        Tuple of (macd_line, signal_line, histogram).
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator (%K and %D).

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        k_period: %K lookback period (default: 14).
        d_period: %D smoothing period (default: 3).

    Returns:
        Tuple of (%K, %D) series.
    """
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()

    denom = highest_high - lowest_low
    k = ((close - lowest_low) / denom.replace(0, float("inf"))) * 100
    d = k.rolling(window=d_period, min_periods=d_period).mean()

    return k, d


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Volume Weighted Average Price.

    Cumulative VWAP from the start of the series.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        volume: Volume series.

    Returns:
        VWAP series.
    """
    typical_price = (high + low + close) / 3
    cum_tp_vol = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    return cum_tp_vol / cum_vol.replace(0, float("inf"))
