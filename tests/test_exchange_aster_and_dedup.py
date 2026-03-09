import pytest

from vibetrading._exchanges.aster import AsterSandbox


class ConcreteAsterSandbox(AsterSandbox):
    def get_supported_assets(self):
        return []

from vibetrading._utils.notification import NotificationDeduplicator


def test_notification_deduplicator_basic(monkeypatch):
    dedup = NotificationDeduplicator(dedup_window_minutes=5)

    err1 = ValueError("HTTP 500 on order 12345")
    assert dedup.should_send("x", "buy", err1) is True
    # Same normalized error within window should not resend
    assert dedup.should_send("x", "buy", ValueError("HTTP 500 on order 99999")) is False

    stats = dedup.get_stats()
    assert stats["recent_unique_errors"] == 1
    assert stats["total_cached_errors"] == 1


def test_aster_sandbox_init_and_stubs():
    sb = ConcreteAsterSandbox(api_key="addr", api_secret="pk", mode="paper", testnet=True)
    assert sb.exchange_name == "aster_testnet"

    with pytest.raises(NotImplementedError):
        sb.get_price("BTC")

    # Convenience wrappers should route through get_price.
    with pytest.raises(NotImplementedError):
        sb.get_spot_price("BTC")
