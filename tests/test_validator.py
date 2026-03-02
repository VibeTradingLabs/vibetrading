"""Tests for the strategy validator."""

from vibetrading._agent.validator import StrategyValidationResult, validate_strategy


class TestValidationResult:
    def test_empty_result_is_valid(self):
        result = StrategyValidationResult()
        assert result.is_valid is True

    def test_result_with_errors_is_invalid(self):
        result = StrategyValidationResult(errors=["something broke"])
        assert result.is_valid is False

    def test_result_with_only_warnings_is_valid(self):
        result = StrategyValidationResult(warnings=["maybe fix this"])
        assert result.is_valid is True

    def test_format_for_llm_valid(self):
        result = StrategyValidationResult()
        output = result.format_for_llm()
        assert "passed" in output.lower()

    def test_format_for_llm_with_errors(self):
        result = StrategyValidationResult(errors=["missing import"])
        output = result.format_for_llm()
        assert "missing import" in output
        assert "ERRORS" in output

    def test_repr_shows_status(self):
        valid = StrategyValidationResult()
        assert "VALID" in repr(valid)

        invalid = StrategyValidationResult(errors=["bad"])
        assert "INVALID" in repr(invalid)


class TestValidateStrategy:
    def test_empty_code_is_invalid(self):
        result = validate_strategy("")
        assert result.is_valid is False
        assert any("empty" in e.lower() for e in result.errors)

    def test_syntax_error_is_caught(self):
        result = validate_strategy("def foo(:\n  pass")
        assert result.is_valid is False
        assert any("syntax" in e.lower() for e in result.errors)

    def test_missing_import_is_error(self):
        code = """
def strategy():
    pass
"""
        result = validate_strategy(code)
        assert result.is_valid is False
        assert any("import" in e.lower() for e in result.errors)

    def test_missing_vibe_decorator_is_error(self):
        code = """
import vibetrading

def strategy():
    pass
"""
        result = validate_strategy(code)
        assert result.is_valid is False
        assert any("@vibe" in e for e in result.errors)

    def test_multiple_vibe_decorators_is_error(self):
        code = """
from vibetrading import vibe

@vibe
def strategy_a():
    pass

@vibe
def strategy_b():
    pass
"""
        result = validate_strategy(code)
        assert result.is_valid is False
        assert any("multiple" in e.lower() or "Multiple" in e for e in result.errors)

    def test_valid_minimal_strategy(self):
        code = """
import vibetrading
from vibetrading import vibe, get_perp_price, set_leverage, long

@vibe
def strategy():
    price = get_perp_price("BTC")
    set_leverage("BTC", 3)
    long("BTC", 0.1, price)
"""
        result = validate_strategy(code)
        # Should have no errors (may have warnings about TP/SL, data validation, etc.)
        assert result.is_valid is True

    def test_missing_set_leverage_is_error(self):
        code = """
from vibetrading import vibe, get_perp_price, long

@vibe
def strategy():
    price = get_perp_price("BTC")
    long("BTC", 0.1, price)
"""
        result = validate_strategy(code)
        assert any("set_leverage" in e for e in result.errors)

    def test_warns_on_missing_data_validation(self):
        code = """
from vibetrading import vibe, get_futures_ohlcv, set_leverage, long

@vibe
def strategy():
    ohlcv = get_futures_ohlcv("BTC", "1h", 50)
    sma = ohlcv["close"].rolling(20).mean().iloc[-1]
    set_leverage("BTC", 3)
    long("BTC", 0.1, 50000)
"""
        result = validate_strategy(code)
        assert any("length" in w.lower() or "len" in w.lower() for w in result.warnings)

    def test_warns_on_missing_nan_check(self):
        code = """
from vibetrading import vibe, get_perp_price, set_leverage, long

@vibe
def strategy():
    price = get_perp_price("BTC")
    set_leverage("BTC", 3)
    long("BTC", 0.1, price)
"""
        result = validate_strategy(code)
        assert any("nan" in w.lower() for w in result.warnings)

    def test_warns_on_hardcoded_balance(self):
        code = """
from vibetrading import vibe, get_perp_price, set_leverage, long

import math

@vibe
def strategy():
    initial_capital = 10000
    price = get_perp_price("BTC")
    if math.isnan(price):
        return
    set_leverage("BTC", 3)
    long("BTC", 0.1, price)
"""
        result = validate_strategy(code)
        assert any("hardcoded" in w.lower() for w in result.warnings)

    def test_warns_on_deprecated_functions(self):
        code = """
from vibetrading import vibe, my_futures_balance, set_leverage, long

@vibe
def strategy():
    balance = my_futures_balance("USDC")
    set_leverage("BTC", 3)
    long("BTC", 0.1, 50000)
"""
        result = validate_strategy(code)
        assert any("deprecated" in w.lower() for w in result.warnings)

    def test_markdown_fences_stripped(self):
        code = """```python
from vibetrading import vibe, get_perp_price, set_leverage, long
import math

@vibe
def strategy():
    price = get_perp_price("BTC")
    if math.isnan(price):
        return
    set_leverage("BTC", 3)
    long("BTC", 0.1, price)
```"""
        result = validate_strategy(code)
        assert result.is_valid is True

    def test_vibetrading_dot_vibe_decorator(self):
        code = """
import vibetrading

@vibetrading.vibe
def strategy():
    price = vibetrading.get_perp_price("BTC")
    vibetrading.set_leverage("BTC", 3)
    vibetrading.long("BTC", 0.1, price)
"""
        result = validate_strategy(code)
        assert result.is_valid is True

    def test_warns_on_no_risk_management(self):
        code = """
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
        result = validate_strategy(code)
        # Should warn about missing reduce_position / TP/SL
        assert any("reduce_position" in w or "TP" in w or "stop" in w.lower() for w in result.warnings)
