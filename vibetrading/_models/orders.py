"""
Pydantic models for orders, positions, and account summaries.

These models define the unified data structures used across all exchange adapters.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class PerpPositionSummary(BaseModel):
    """Per-asset perpetual position information."""

    asset: str
    size: float
    entry_price: float
    unrealized_pnl: float
    position_value: float
    margin_used: float
    liquidation_price: Optional[float] = None
    funding: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class PerpAccountSummary(BaseModel):
    """Account-level perpetual summary fields."""
    time: Optional[int] = None
    account_value: float
    available_margin: float
    total_margin_used: float
    total_unrealized_pnl: float
    positions: List[PerpPositionSummary] = []

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class SpotBalanceSummary(BaseModel):
    """Per-asset spot balance information."""
    asset: str
    total: float
    free: float
    locked: float

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class SpotAccountSummary(BaseModel):
    """Top-level spot summary structure."""
    time: Optional[int] = None
    balances: List[SpotBalanceSummary] = []

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class SpotOrder(BaseModel):
    """Spot order model."""
    id: str
    client_id: Optional[str] = None
    asset: str
    side: str
    type: str
    size: float
    price: Optional[float] = None
    timestamp: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class SpotOrderResponse(BaseModel):
    """Response model for spot order operations."""
    status: str
    error: Optional[str] = None
    order: Optional[SpotOrder] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @staticmethod
    def Error(error: str) -> 'SpotOrderResponse':
        return SpotOrderResponse(status="error", error=error, order=None)


class PerpOrder(BaseModel):
    """Perpetual order model."""
    id: str
    client_id: Optional[str] = None
    asset: str
    side: str
    type: str
    size: float
    leverage: Optional[float] = None
    price: Optional[float] = None
    reduce_only: Optional[bool] = None
    timestamp: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class PerpOrderResponse(BaseModel):
    """Response model for perpetual order operations."""
    status: str
    error: Optional[str] = None
    order: Optional[PerpOrder] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @staticmethod
    def Error(error: str) -> 'PerpOrderResponse':
        return PerpOrderResponse(status="error", error=error, order=None)


class CancelOrdersResponse(BaseModel):
    """Response model for batch order cancellation."""
    status: str
    error: Optional[str] = None
    orders: List[Dict[str, Any]] = []

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @staticmethod
    def Error(error: str) -> 'CancelOrdersResponse':
        return CancelOrdersResponse(status="error", error=error, orders=[])
