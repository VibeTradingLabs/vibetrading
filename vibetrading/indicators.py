"""
Built-in technical indicators — no external dependencies required.

These are pure-pandas implementations suitable for use inside @vibe strategy
functions or standalone analysis.

Usage::

    from vibetrading.indicators import rsi, sma, ema, bbands, atr, macd, stochastic, vwap

    ohlcv = get_futures_ohlcv("BTC", "1h", 50)
    rsi_14 = rsi(ohlcv["close"])
    upper, middle, lower = bbands(ohlcv["close"])
    macd_line, signal, hist = macd(ohlcv["close"])
    atr_14 = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"])
"""

from ._utils.indicators import (
    atr,
    bbands,
    ema,
    macd,
    rsi,
    sma,
    stochastic,
    vwap,
)

__all__ = [
    "atr",
    "bbands",
    "ema",
    "macd",
    "rsi",
    "sma",
    "stochastic",
    "vwap",
]
