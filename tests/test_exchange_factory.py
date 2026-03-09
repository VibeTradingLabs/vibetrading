import pytest

from vibetrading._exchanges import create_sandbox


def test_create_sandbox_unsupported_exchange():
    with pytest.raises(ValueError):
        create_sandbox("nope")


def test_create_sandbox_hyperliquid_missing_sdk():
    # In CI/dev installs we don't install vibetrading[hyperliquid] by default.
    # The factory should surface an ImportError from the sandbox.
    with pytest.raises(ImportError):
        create_sandbox("hyperliquid", api_key="x", api_secret="y")
