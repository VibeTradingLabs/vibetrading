"""
Notification deduplicator for preventing duplicate error notifications.

Each sandbox instance maintains its own deduplication dictionary,
auto-refreshing after a configurable window (default 5 minutes).
"""

import re
from datetime import datetime, timezone, timedelta
from typing import Dict


class NotificationDeduplicator:
    """
    Error notification deduplicator.

    Maintains a per-instance dictionary of recently sent error types.
    The same error type will only be notified once per dedup_window_minutes.
    """

    def __init__(self, dedup_window_minutes: int = 5):
        self.dedup_window = timedelta(minutes=dedup_window_minutes)
        self.sent_errors: Dict[str, datetime] = {}

    def should_send(self, exchange: str, operation: str, error: Exception) -> bool:
        """
        Determine whether a notification should be sent.

        Args:
            exchange: Exchange name (e.g. 'hyperliquid', 'paradex')
            operation: Operation type (e.g. 'long', 'short', 'buy')
            error: The exception object

        Returns:
            True if notification should be sent, False if recently sent
        """
        error_type = type(error).__name__
        error_message = str(error).strip()
        normalized_error = self._normalize_error_message(error_message)
        error_key = f"{exchange}:{operation}:{error_type}:{normalized_error}"

        now = datetime.now(timezone.utc)

        if error_key in self.sent_errors:
            last_sent_time = self.sent_errors[error_key]
            if now - last_sent_time < self.dedup_window:
                return False

        self.sent_errors[error_key] = now
        self._cleanup_old_entries(now)
        return True

    def _normalize_error_message(self, error_msg: str) -> str:
        """Normalize error message by removing variable parts (numbers, UUIDs, etc.)."""
        if not error_msg:
            return ""
        normalized = error_msg[:200]
        normalized = re.sub(r'\b\d+\.\d+\b', '', normalized)
        normalized = re.sub(r'\b\d{4,}\b', '', normalized)
        normalized = re.sub(
            r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}',
            '', normalized, flags=re.IGNORECASE
        )
        normalized = re.sub(r'0x[a-f0-9]+', '', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized

    def _cleanup_old_entries(self, now: datetime):
        """Remove entries older than the dedup window."""
        cutoff_time = now - self.dedup_window
        keys_to_remove = [k for k, v in self.sent_errors.items() if v < cutoff_time]
        for k in keys_to_remove:
            del self.sent_errors[k]

    def get_stats(self) -> Dict[str, int]:
        """Get deduplication statistics."""
        now = datetime.now(timezone.utc)
        cutoff_time = now - self.dedup_window
        recent_count = sum(1 for v in self.sent_errors.values() if v >= cutoff_time)
        return {
            "recent_unique_errors": recent_count,
            "total_cached_errors": len(self.sent_errors)
        }
