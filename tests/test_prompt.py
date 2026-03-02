"""Tests for prompt building and templates."""

from vibetrading._agent.prompt import (
    STRATEGY_SYSTEM_PROMPT,
    VIBETRADING_API_REFERENCE,
    STRATEGY_CONSTRAINTS,
    build_generation_prompt,
)


class TestPromptConstants:
    def test_system_prompt_not_empty(self):
        assert len(STRATEGY_SYSTEM_PROMPT) > 100

    def test_api_reference_contains_key_functions(self):
        assert "get_perp_price" in VIBETRADING_API_REFERENCE
        assert "get_perp_summary" in VIBETRADING_API_REFERENCE
        assert "@vibe" in VIBETRADING_API_REFERENCE
        assert "set_leverage" in VIBETRADING_API_REFERENCE
        assert "reduce_position" in VIBETRADING_API_REFERENCE

    def test_constraints_contain_rules(self):
        assert "set_leverage" in STRATEGY_CONSTRAINTS
        assert "USDC" in STRATEGY_CONSTRAINTS
        assert "@vibe" in STRATEGY_CONSTRAINTS


class TestBuildGenerationPrompt:
    def test_basic_prompt(self):
        messages = build_generation_prompt("BTC momentum strategy")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "BTC momentum strategy" in messages[1]["content"]

    def test_prompt_with_assets(self):
        messages = build_generation_prompt("momentum", assets=["BTC", "ETH"])
        user_msg = messages[1]["content"]
        assert "BTC" in user_msg
        assert "ETH" in user_msg

    def test_prompt_with_market_type(self):
        messages = build_generation_prompt("grid strategy", market_type="perp")
        user_msg = messages[1]["content"]
        assert "perp" in user_msg

    def test_prompt_with_max_leverage(self):
        messages = build_generation_prompt("scalping", max_leverage=5)
        user_msg = messages[1]["content"]
        assert "5" in user_msg

    def test_prompt_with_interval(self):
        messages = build_generation_prompt("mean reversion", interval="1h")
        user_msg = messages[1]["content"]
        assert "1h" in user_msg

    def test_prompt_with_additional_context(self):
        messages = build_generation_prompt(
            "RSI strategy",
            additional_context="Previous attempt had too many trades",
        )
        user_msg = messages[1]["content"]
        assert "Previous attempt" in user_msg

    def test_system_prompt_contains_api_reference(self):
        messages = build_generation_prompt("any strategy")
        system = messages[0]["content"]
        assert "get_perp_price" in system
        assert "@vibe" in system
