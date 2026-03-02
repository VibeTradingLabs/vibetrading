"""
Pydantic models for orders, positions, and account summaries.

These models define the unified data structures used across all exchange adapters.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class PerpPositionSummary(BaseModel):
    """Per-asset perpetual position information."""

    asset: str
    size: float
    entry_price: float
    unrealized_pnl: float
    position_value: float
    margin_used: float
    liquidation_price: float | None = None
    funding: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class PerpAccountSummary(BaseModel):
    """Account-level perpetual summary fields."""

    time: int | None = None
    account_value: float
    available_margin: float
    total_margin_used: float
    total_unrealized_pnl: float
    positions: list[PerpPositionSummary] = []

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class SpotBalanceSummary(BaseModel):
    """Per-asset spot balance information."""

    asset: str
    total: float
    free: float
    locked: float

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class SpotAccountSummary(BaseModel):
    """Top-level spot summary structure."""

    time: int | None = None
    balances: list[SpotBalanceSummary] = []

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class SpotOrder(BaseModel):
    """Spot order model."""

    id: str
    client_id: str | None = None
    asset: str
    side: str
    type: str
    size: float
    price: float | None = None
    timestamp: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class SpotOrderResponse(BaseModel):
    """Response model for spot order operations."""

    status: str
    error: str | None = None
    order: SpotOrder | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @staticmethod
    def Error(error: str) -> SpotOrderResponse:
        return SpotOrderResponse(status="error", error=error, order=None)


class PerpOrder(BaseModel):
    """Perpetual order model."""

    id: str
    client_id: str | None = None
    asset: str
    side: str
    type: str
    size: float
    leverage: float | None = None
    price: float | None = None
    reduce_only: bool | None = None
    timestamp: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class PerpOrderResponse(BaseModel):
    """Response model for perpetual order operations."""

    status: str
    error: str | None = None
    order: PerpOrder | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @staticmethod
    def Error(error: str) -> PerpOrderResponse:
        return PerpOrderResponse(status="error", error=error, order=None)


class CancelOrdersResponse(BaseModel):
    """Response model for batch order cancellation."""

    status: str
    error: str | None = None
    orders: list[dict[str, Any]] = []

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @staticmethod
    def Error(error: str) -> CancelOrdersResponse:
        return CancelOrdersResponse(status="error", error=error, orders=[])
