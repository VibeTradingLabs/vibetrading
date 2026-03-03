"""Tests for the CLI module."""

import argparse
from unittest import mock

from vibetrading.cli import cmd_validate, cmd_version, main


class TestCLIVersion:
    def test_version_returns_zero(self):
        assert cmd_version(argparse.Namespace()) == 0


class TestCLIValidate:
    def test_validate_missing_file(self, tmp_path):
        args = argparse.Namespace(strategy=str(tmp_path / "nonexistent.py"))
        assert cmd_validate(args) == 1

    def test_validate_valid_strategy(self, tmp_path):
        strategy = tmp_path / "valid.py"
        strategy.write_text(
            """
import vibetrading
from vibetrading import vibe, get_perp_price, set_leverage, long
import math

@vibe
def strategy():
    price = get_perp_price("BTC")
    if math.isnan(price):
        return
    set_leverage("BTC", 3)
    long("BTC", 0.1, price)
"""
        )
        args = argparse.Namespace(strategy=str(strategy))
        assert cmd_validate(args) == 0

    def test_validate_invalid_strategy(self, tmp_path):
        strategy = tmp_path / "invalid.py"
        strategy.write_text("x = 1")
        args = argparse.Namespace(strategy=str(strategy))
        assert cmd_validate(args) == 1


class TestCLIMain:
    def test_no_args_returns_zero(self):
        with mock.patch("sys.argv", ["vibetrading"]):
            assert main() == 0

    def test_version_command(self):
        with mock.patch("sys.argv", ["vibetrading", "version"]):
            assert main() == 0
