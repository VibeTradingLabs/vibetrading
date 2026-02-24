"""
Common base helpers shared across live exchange sandbox implementations.
"""

import logging
from typing import Dict, List, Optional, Any

from .._core.sandbox_base import VibeSandboxBase, SUPPORTED_INTERVALS, SUPPORTED_LEVERAGE
from .._models.types import SpotMeta, PerpMeta
from .._utils.math import truncate_quantity
from .._utils.notification import NotificationDeduplicator
logger = logging.getLogger(__name__)


class LiveSandboxBase(VibeSandboxBase):
    """
    Convenience base class for live exchange sandboxes.

    Provides:
    - Notification deduplication
    - Common state attributes
    """

    def __init__(
        self,
        exchange_name: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        mode: str = "live",
        notification_deduplicator: Optional[NotificationDeduplicator] = None,
        **kwargs,
    ):
        self.exchange_name = exchange_name.lower()
        self.api_key = api_key
        self.api_secret = api_secret
        self.mode = mode
        self.fee_rate = 0.001

        self.spot_meta: Dict[str, SpotMeta] = {}
        self.perp_meta: Dict[str, PerpMeta] = {}
        self.asset_spot_mapping: Dict[str, str] = {}
        self.spot_asset_mapping: Dict[str, str] = {}

        self.notification_deduplicator = notification_deduplicator or NotificationDeduplicator()

        from datetime import datetime, timezone
        self.current_time = datetime.now(timezone.utc)

    def get_current_time(self):
        from datetime import datetime, timezone
        self.current_time = datetime.now(timezone.utc)
        return self.current_time
