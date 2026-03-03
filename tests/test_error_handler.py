"""Tests for StrategyErrorHandler."""

from datetime import datetime, timezone

import pytest

from vibetrading._core.error_handler import StrategyErrorHandler


class _FakeSandbox:
    current_time = datetime(2025, 3, 1, tzinfo=timezone.utc)


class TestStrategyErrorHandler:
    def test_wrap_strategy_passes_through(self):
        handler = StrategyErrorHandler(_FakeSandbox(), "pass")
        calls = []

        def my_strategy():
            calls.append(1)
            return 42

        wrapped = handler.wrap_strategy(my_strategy)
        result = wrapped()
        assert result == 42
        assert len(calls) == 1

    def test_wrap_strategy_reraises_error(self, capsys):
        handler = StrategyErrorHandler(_FakeSandbox(), "x = 1/0")

        def bad_strategy():
            raise ValueError("test error")

        wrapped = handler.wrap_strategy(bad_strategy)
        with pytest.raises(ValueError, match="test error"):
            wrapped()

        output = capsys.readouterr().out
        assert "Strategy Error" in output
        assert "test error" in output

    def test_extract_error_context_no_traceback(self):
        handler = StrategyErrorHandler(_FakeSandbox(), "print('hello')")
        error = ValueError("no traceback")
        # Without a real traceback, should return None
        result = handler._extract_error_context(error)
        assert result is None

    def test_print_error_context(self, capsys):
        handler = StrategyErrorHandler(_FakeSandbox(), "")
        context = {
            "error_frames": [
                {"line_no": 5, "function": "strategy", "code": "x = 1/0"},
            ],
            "primary_error_line": 5,
            "primary_error_code": "x = 1/0",
        }
        handler._print_error_context(context)
        output = capsys.readouterr().out
        assert "Line 5" in output
        assert "x = 1/0" in output

    def test_handle_error_logs_details(self, capsys):
        handler = StrategyErrorHandler(_FakeSandbox(), "test code")
        try:
            raise TypeError("bad type")
        except TypeError as e:
            handler._handle_error(e, "test_func")

        output = capsys.readouterr().out
        assert "TypeError" in output
        assert "bad type" in output
        assert "test_func" in output
