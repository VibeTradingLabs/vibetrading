"""Tests that all sample strategies pass validation."""

from pathlib import Path

import pytest

from vibetrading._agent.validator import validate_strategy

STRATEGIES_DIR = Path(__file__).parent.parent / "strategies"


def _get_strategy_files():
    """Collect all .py strategy files."""
    if not STRATEGIES_DIR.exists():
        return []
    return sorted(STRATEGIES_DIR.glob("*.py"))


@pytest.mark.parametrize(
    "strategy_file",
    _get_strategy_files(),
    ids=lambda f: f.stem,
)
class TestSampleStrategies:
    def test_strategy_is_valid(self, strategy_file):
        """Every sample strategy must pass static validation."""
        code = strategy_file.read_text()
        result = validate_strategy(code)
        assert result.is_valid, f"{strategy_file.name} failed validation:\n" + "\n".join(
            f"  ERROR: {e}" for e in result.errors
        )

    def test_strategy_has_docstring(self, strategy_file):
        """Every sample strategy should have a module docstring."""
        code = strategy_file.read_text()
        assert code.strip().startswith('"""') or code.strip().startswith("'''"), (
            f"{strategy_file.name} is missing a module docstring"
        )

    def test_strategy_has_vibe_decorator(self, strategy_file):
        """Every sample strategy must use the @vibe decorator."""
        code = strategy_file.read_text()
        assert "@vibe" in code, f"{strategy_file.name} is missing @vibe decorator"

    def test_strategy_imports_vibetrading(self, strategy_file):
        """Every sample strategy must import from vibetrading."""
        code = strategy_file.read_text()
        assert "vibetrading" in code, f"{strategy_file.name} doesn't import vibetrading"
