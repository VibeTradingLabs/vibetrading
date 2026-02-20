"""
VibeSandboxBase - Abstract base class for trading strategy sandbox environment.

This module provides the interface for all exchange adapters and the backtesting
sandbox. Every exchange implementation and the StaticSandbox (backtesting)
must implement this interface, ensuring strategy code runs identically across
live trading and backtesting.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

SUPPORTED_INTERVALS = {
    "1s": timedelta(seconds=1),
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "6h": timedelta(hours=6),
    "1d": timedelta(days=1),
}

SUPPORTED_LEVERAGE = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


class VibeSandboxBase(ABC):
    """Abstract base class for VibeSandbox implementations."""

    # ── Asset Support ──────────────────────────────────────────────────
    @abstractmethod
    def get_supported_assets(self) -> List[str]:
        """Get list of supported assets for trading."""
        pass

    @abstractmethod
    def my_spot_balance(self, asset: str) -> float:
        """Get the current spot balance for an asset."""
        pass

    @abstractmethod
    def my_futures_balance(self, asset: str) -> float:
        """Get the current futures balance for an asset."""
        pass

    # ── Order Management ───────────────────────────────────────────────
    @abstractmethod
    def get_perp_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Get open perpetual orders.

        Returns:
            List of dicts each containing: id, client_id, asset, side, type,
            size, leverage, price, reduce_only, timestamp
        """
        pass

    @abstractmethod
    def get_spot_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Get open spot orders.

        Returns:
            List of dicts each containing: id, client_id, asset, side, type,
            size, price, timestamp
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: int) -> None:
        """Cancel an order by its ID."""
        pass

    @abstractmethod
    def cancel_spot_orders(self, asset: str, order_ids: List[str]) -> Dict[str, Any]:
        """Cancel multiple spot orders by their IDs (max 10 per request)."""
        pass

    @abstractmethod
    def cancel_perp_orders(self, asset: str, order_ids: List[str]) -> Dict[str, Any]:
        """Cancel multiple perp orders by their IDs (max 10 per request)."""
        pass

    # ── Trading Operations ─────────────────────────────────────────────
    @abstractmethod
    def set_leverage(self, asset: str, leverage: int) -> None:
        """Set the leverage for an asset."""
        pass

    @abstractmethod
    def buy(self, asset: str, quantity: float, price: float, order_type: str = "limit") -> Dict[str, Any]:
        """Execute a spot buy order.

        Returns:
            Dict with: status, error, order (SpotOrder fields)
        """
        pass

    @abstractmethod
    def sell(self, asset: str, quantity: float, price: float, order_type: str = "limit") -> Dict[str, Any]:
        """Execute a spot sell order."""
        pass

    @abstractmethod
    def long(self, asset: str, quantity: float, price: float, order_type: str = "limit") -> Dict[str, Any]:
        """Execute a futures long position.

        Returns:
            Dict with: status, error, order (PerpOrder fields)
        """
        pass

    @abstractmethod
    def short(self, asset: str, quantity: float, price: float, order_type: str = "limit") -> Dict[str, Any]:
        """Execute a futures short position."""
        pass

    @abstractmethod
    def reduce_position(self, asset: str, quantity: float) -> Dict[str, Any]:
        """Reduce a futures position for an asset."""
        pass

    @abstractmethod
    def get_futures_position(self, asset: str) -> float:
        """Get current futures position size (positive=long, negative=short)."""
        pass

    # ── Market Data ────────────────────────────────────────────────────
    @abstractmethod
    def get_price(self, asset: str) -> float:
        """Get the current price for an asset."""
        pass

    @abstractmethod
    def get_spot_price(self, asset: str) -> float:
        """Get the current spot price for an asset."""
        pass

    @abstractmethod
    def get_perp_price(self, asset: str) -> float:
        """Get the current perp price for an asset."""
        pass

    @abstractmethod
    def get_spot_ohlcv(self, asset: str, interval: str, limit: int) -> pd.DataFrame:
        """Get spot OHLCV data. Returns DataFrame with [open, high, low, close, volume]."""
        pass

    @abstractmethod
    def get_futures_ohlcv(self, asset: str, interval: str, limit: int) -> pd.DataFrame:
        """Get futures OHLCV data. Returns DataFrame with [open, high, low, close, volume, fundingRate, openInterest]."""
        pass

    @abstractmethod
    def get_funding_rate(self, asset: str) -> float:
        """Get the current funding rate for an asset."""
        pass

    @abstractmethod
    def get_funding_rate_history(self, asset: str, limit: int) -> pd.DataFrame:
        """Get historical funding rate data for an asset."""
        pass

    @abstractmethod
    def get_open_interest(self, asset: str) -> float:
        """Get the current open interest for an asset."""
        pass

    @abstractmethod
    def get_open_interest_history(self, asset: str, limit: int) -> pd.DataFrame:
        """Get historical open interest data for an asset."""
        pass

    # ── Account Summaries ──────────────────────────────────────────────
    @abstractmethod
    def get_perp_summary(self) -> Dict[str, Any]:
        """Get account-level perpetual summary with positions."""
        pass

    @abstractmethod
    def get_perp_position(self, asset: str) -> Optional[Dict[str, Any]]:
        """Get the current perp position for an asset."""
        pass

    @abstractmethod
    def get_spot_summary(self) -> Dict[str, Any]:
        """Get spot account summary with balances."""
        pass

    # ── Time ───────────────────────────────────────────────────────────
    @abstractmethod
    def get_current_time(self) -> datetime:
        """Get the current simulation / wall-clock time."""
        pass

    # ── Non-abstract helpers ───────────────────────────────────────────
    def advance_time(self, interval: str) -> Optional[datetime]:
        """Advance the simulation time by the specified interval."""
        return None

    def get_interval_minutes(self, interval: str) -> int:
        """Get the interval in minutes."""
        mapping = {
            "1s": 1 / 60,
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
            "1w": 10080,
            "1M": 43200,
        }
        if interval not in mapping:
            raise ValueError(f"Unsupported interval: {interval}")
        return mapping[interval]
