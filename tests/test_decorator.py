"""Tests for the @vibe decorator."""

from vibetrading._core.decorator import vibe


class TestVibeDecorator:
    def test_bare_decorator(self):
        @vibe
        def strategy():
            pass

        assert strategy._is_vibe_strategy is True
        assert strategy._vibe_interval == "1m"

    def test_empty_parens_decorator(self):
        @vibe()
        def strategy():
            pass

        assert strategy._is_vibe_strategy is True
        assert strategy._vibe_interval == "1m"

    def test_decorator_with_interval(self):
        @vibe(interval="1h")
        def strategy():
            pass

        assert strategy._is_vibe_strategy is True
        assert strategy._vibe_interval == "1h"

    def test_decorator_with_positional_interval(self):
        @vibe("5m")
        def strategy():
            pass

        assert strategy._is_vibe_strategy is True
        assert strategy._vibe_interval == "5m"

    def test_decorated_function_still_callable(self):
        @vibe
        def strategy():
            return 42

        assert strategy() == 42

    def test_function_name_preserved(self):
        @vibe
        def my_strategy():
            pass

        assert my_strategy.__name__ == "my_strategy"
