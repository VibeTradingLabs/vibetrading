"""
Paradex exchange sandbox for live trading on StarkNet.

Requires: pip install vibetrading[paradex]
  (starknet-py, requests, websockets)
"""

import logging
from typing import Dict, List, Any, Optional

import pandas as pd

from .base import LiveSandboxBase
from ..models.orders import (
    PerpAccountSummary, PerpPositionSummary, SpotAccountSummary,
    CancelOrdersResponse, PerpOrderResponse, PerpOrder,
)
from ..utils.notification import NotificationDeduplicator

logger = logging.getLogger(__name__)


class ParadexSandbox(LiveSandboxBase):
    """Live trading sandbox for Paradex on StarkNet (perps + JWT auth)."""

    BASE_URL = "https://api.prod.paradex.trade"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        mode: str = "live",
        account_address: Optional[str] = None,
        private_key: Optional[str] = None,
        notification_deduplicator: Optional[NotificationDeduplicator] = None,
        **kwargs,
    ):
        super().__init__(
            exchange_name="paradex",
            api_key=api_key or account_address,
            api_secret=api_secret or private_key,
            mode=mode,
            notification_deduplicator=notification_deduplicator,
            **kwargs,
        )
        self.account_address = account_address or api_key
        self.private_key = private_key or api_secret
        logger.info("ParadexSandbox ready (mode=%s)", mode)

    # -- Stub implementations (same pattern as Extended) ---------------
    def get_price(self, asset):
        raise NotImplementedError

    def get_spot_price(self, asset):
        raise NotImplementedError("Spot not supported on Paradex")

    def get_perp_price(self, asset):
        return self.get_price(asset)

    def my_spot_balance(self, asset):
        return 0.0

    def my_futures_balance(self, asset="USDC"):
        raise NotImplementedError

    def buy(self, asset, quantity, price, order_type="limit"):
        raise NotImplementedError("Spot not supported on Paradex")

    def sell(self, asset, quantity, price, order_type="limit"):
        raise NotImplementedError("Spot not supported on Paradex")

    def long(self, asset, quantity, price, order_type="limit"):
        raise NotImplementedError

    def short(self, asset, quantity, price, order_type="limit"):
        raise NotImplementedError

    def reduce_position(self, asset, quantity):
        raise NotImplementedError

    def set_leverage(self, asset, leverage):
        raise NotImplementedError

    def get_futures_position(self, asset):
        raise NotImplementedError

    def get_spot_ohlcv(self, asset, interval, limit):
        return pd.DataFrame()

    def get_futures_ohlcv(self, asset, interval, limit):
        raise NotImplementedError

    def get_funding_rate(self, asset="BTC", timestamp=None):
        return 0.0

    def get_funding_rate_history(self, asset, limit):
        return pd.DataFrame(columns=["timestamp", "fundingRate"])

    def get_open_interest(self, asset="BTC", timestamp=None):
        return 0.0

    def get_open_interest_history(self, asset, limit):
        return pd.DataFrame(columns=["timestamp", "openInterest"])

    def cancel_order(self, order_id):
        raise NotImplementedError

    def get_perp_open_orders(self, asset=None):
        return []

    def get_spot_open_orders(self, asset=None):
        return []

    def cancel_spot_orders(self, asset, order_ids):
        return CancelOrdersResponse(status="success", orders=[]).to_dict()

    def cancel_perp_orders(self, asset, order_ids):
        raise NotImplementedError

    def get_perp_summary(self):
        return PerpAccountSummary().to_dict()

    def get_perp_position(self, asset):
        return None

    def get_spot_summary(self):
        return SpotAccountSummary().to_dict()

    def get_futures_unrealized_pnl(self, asset=None):
        return 0.0

    def get_total_futures_unrealized_pnl(self):
        return 0.0

    def get_available_margin(self):
        return 0.0

    def get_total_margin_used(self):
        return 0.0

    def apply_funding_payment(self, asset, position_size, leverage=1):
        pass

    def advance_time(self, interval):
        return None
