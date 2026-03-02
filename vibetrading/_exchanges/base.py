"""
Common base helpers shared across live exchange sandbox implementations.
"""

import logging

from .._core.sandbox_base import VibeSandboxBase
from .._models.types import PerpMeta, SpotMeta
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
        api_key: str | None = None,
        api_secret: str | None = None,
        mode: str = "live",
        notification_deduplicator: NotificationDeduplicator | None = None,
        **kwargs,
    ):
        self.exchange_name = exchange_name.lower()
        self.api_key = api_key
        self.api_secret = api_secret
        self.mode = mode
        self.fee_rate = 0.001

        self.spot_meta: dict[str, SpotMeta] = {}
        self.perp_meta: dict[str, PerpMeta] = {}
        self.asset_spot_mapping: dict[str, str] = {}
        self.spot_asset_mapping: dict[str, str] = {}

        self.notification_deduplicator = notification_deduplicator or NotificationDeduplicator()

        from datetime import datetime, timezone

        self.current_time = datetime.now(timezone.utc)

    def get_current_time(self):
        from datetime import datetime, timezone

        self.current_time = datetime.now(timezone.utc)
        return self.current_time
