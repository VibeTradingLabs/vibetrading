"""
Data models for orders, positions, and exchange metadata.

Usage::

    import vibetrading.models

    order = vibetrading.models.SpotOrder(
        id="1", asset="BTC", side="buy", type="limit",
        size=0.1, price=68000.0, timestamp=1700000000,
    )
"""

from ._models.orders import (
    PerpAccountSummary,
    PerpPositionSummary,
    SpotAccountSummary,
    SpotBalanceSummary,
    SpotOrder,
    SpotOrderResponse,
    PerpOrder,
    PerpOrderResponse,
    CancelOrdersResponse,
)
from ._models.types import SpotMeta, PerpMeta, AgentMetadata

__all__ = [
    "PerpAccountSummary",
    "PerpPositionSummary",
    "SpotAccountSummary",
    "SpotBalanceSummary",
    "SpotOrder",
    "SpotOrderResponse",
    "PerpOrder",
    "PerpOrderResponse",
    "CancelOrdersResponse",
    "SpotMeta",
    "PerpMeta",
    "AgentMetadata",
]
