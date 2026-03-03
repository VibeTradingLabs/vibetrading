"""Tests for strategy templates."""

import pytest

from vibetrading._agent.validator import validate_strategy
from vibetrading.templates import (
    dca,
    get_template,
    grid,
    list_templates,
    mean_reversion,
    momentum,
    multi_momentum,
)


class TestTemplateRegistry:
    def test_list_templates(self):
        templates = list_templates()
        assert "momentum" in templates
        assert "mean_reversion" in templates
        assert "grid" in templates
        assert "dca" in templates

    def test_get_template(self):
        t = get_template("momentum")
        assert hasattr(t, "generate")
        assert hasattr(t, "DEFAULTS")

    def test_get_template_not_found(self):
        with pytest.raises(KeyError, match="not found"):
            get_template("nonexistent")


class TestMomentumTemplate:
    def test_generate_default(self):
        code = momentum.generate()
        assert "@vibe" in code
        assert "get_perp_price" in code
        assert "set_leverage" in code
        assert "reduce_position" in code
        assert 'ASSET = "BTC"' in code

    def test_generate_custom_params(self):
        code = momentum.generate(asset="ETH", leverage=5, sma_fast=7)
        assert 'ASSET = "ETH"' in code
        assert "LEVERAGE = 5" in code
        assert "SMA_FAST = 7" in code

    def test_generated_code_validates(self):
        code = momentum.generate()
        result = validate_strategy(code)
        assert result.is_valid, f"Validation failed: {result.errors}"


class TestMeanReversionTemplate:
    def test_generate_default(self):
        code = mean_reversion.generate()
        assert "@vibe" in code
        assert "BB_PERIOD" in code
        assert "lower_band" in code

    def test_generate_custom_params(self):
        code = mean_reversion.generate(asset="SOL", bb_period=30, bb_std=2.5)
        assert 'ASSET = "SOL"' in code
        assert "BB_PERIOD = 30" in code
        assert "BB_STD = 2.5" in code

    def test_generated_code_validates(self):
        code = mean_reversion.generate()
        result = validate_strategy(code)
        assert result.is_valid, f"Validation failed: {result.errors}"


class TestGridTemplate:
    def test_generate_default(self):
        code = grid.generate()
        assert "@vibe" in code
        assert "GRID_LEVELS" in code
        assert "buy(" in code
        assert "sell(" in code

    def test_generate_custom_params(self):
        code = grid.generate(asset="ETH", grid_levels=10, grid_spacing_pct=0.01)
        assert 'ASSET = "ETH"' in code
        assert "GRID_LEVELS = 10" in code

    def test_generated_code_validates(self):
        code = grid.generate()
        result = validate_strategy(code)
        # Grid template uses spot trading, may have different warnings
        # but should not have hard errors
        assert result.is_valid, f"Validation failed: {result.errors}"


class TestDCATemplate:
    def test_generate_default(self):
        code = dca.generate()
        assert "@vibe" in code
        assert "BUY_AMOUNT" in code
        assert "buy(" in code

    def test_generate_custom_params(self):
        code = dca.generate(asset="ETH", buy_amount=100, tp_pct=0.20)
        assert 'ASSET = "ETH"' in code
        assert "BUY_AMOUNT = 100" in code
        assert "TP_PCT = 0.2" in code

    def test_generated_code_validates(self):
        code = dca.generate()
        result = validate_strategy(code)
        assert result.is_valid, f"Validation failed: {result.errors}"


class TestMultiMomentumTemplate:
    def test_generate_default(self):
        code = multi_momentum.generate()
        assert "@vibe" in code
        assert "ASSETS" in code
        assert "for asset in ASSETS" in code

    def test_generate_custom_assets(self):
        code = multi_momentum.generate(assets=["BTC", "ETH", "SOL", "AVAX"])
        assert "AVAX" in code

    def test_generate_custom_params(self):
        code = multi_momentum.generate(leverage=5, tp_pct=0.05)
        assert "LEVERAGE = 5" in code
        assert "TP_PCT = 0.05" in code

    def test_generated_code_validates(self):
        code = multi_momentum.generate()
        result = validate_strategy(code)
        assert result.is_valid, f"Validation failed: {result.errors}"

    def test_list_includes_multi_momentum(self):
        assert "multi_momentum" in list_templates()
