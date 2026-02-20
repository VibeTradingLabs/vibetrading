"""
Strategy code validator for LLM-generated strategies.

Performs static analysis on generated strategy code to catch common errors
before execution. This enables a closed-loop workflow where validation
errors are fed back to the LLM for automatic correction.

Usage::

    from vibetrading.agent import validate_strategy

    result = validate_strategy(code)
    if not result.is_valid:
        for error in result.errors:
            print(f"Error: {error}")
        for warning in result.warnings:
            print(f"Warning: {warning}")
"""

import ast
import re
from dataclasses import dataclass, field


@dataclass
class StrategyValidationResult:
    """Result of strategy code validation.

    Attributes:
        is_valid: True if no errors were found (warnings are acceptable).
        errors: List of error messages that must be fixed.
        warnings: List of warning messages (non-blocking but recommended to fix).
    """
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def __repr__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        parts = [f"StrategyValidationResult({status})"]
        for e in self.errors:
            parts.append(f"  ERROR: {e}")
        for w in self.warnings:
            parts.append(f"  WARNING: {w}")
        return "\n".join(parts)

    def format_for_llm(self) -> str:
        """Format validation results as feedback for an LLM to fix.

        Returns a string suitable for appending to a follow-up prompt
        so the LLM can correct the generated code.
        """
        if self.is_valid and not self.warnings:
            return "Strategy code passed all validation checks."

        parts = ["The generated strategy code has the following issues:\n"]
        if self.errors:
            parts.append("ERRORS (must fix):")
            for i, e in enumerate(self.errors, 1):
                parts.append(f"  {i}. {e}")
        if self.warnings:
            parts.append("\nWARNINGS (recommended to fix):")
            for i, w in enumerate(self.warnings, 1):
                parts.append(f"  {i}. {w}")
        parts.append("\nPlease regenerate the strategy code fixing all errors.")
        return "\n".join(parts)


def validate_strategy(code: str) -> StrategyValidationResult:
    """
    Validate generated strategy code against vibetrading conventions.

    Performs static analysis to check for:
    - Correct imports and decorator usage
    - Data validation patterns
    - Risk management practices (TP/SL)
    - Leverage and balance usage
    - Common LLM generation mistakes

    Args:
        code: Python source code of the strategy.

    Returns:
        StrategyValidationResult with errors and warnings.

    Example::

        result = validate_strategy(strategy_code)
        if not result.is_valid:
            # Feed errors back to LLM
            feedback = result.format_for_llm()
            messages.append({"role": "user", "content": feedback})
    """
    result = StrategyValidationResult()

    if not code or not code.strip():
        result.errors.append("Strategy code is empty.")
        return result

    # Strip markdown code fences if present
    code = _strip_markdown_fences(code)

    # Check syntax
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        result.errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        return result

    _check_imports(code, tree, result)
    _check_vibe_decorator(code, tree, result)
    _check_leverage(code, result)
    _check_data_validation(code, result)
    _check_position_sizing(code, result)
    _check_risk_management(code, result)
    _check_nan_handling(code, result)
    _check_hardcoded_balance(code, result)
    _check_deprecated_functions(code, result)

    return result


def _strip_markdown_fences(code: str) -> str:
    """Remove markdown code fences from LLM output."""
    code = code.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        # Remove opening fence (```python, ```py, ```)
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        # Remove closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)
    return code


def _check_imports(code: str, tree: ast.Module, result: StrategyValidationResult):
    """Check that vibetrading is imported correctly."""
    has_import = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "vibetrading":
                    has_import = True
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("vibetrading"):
                has_import = True

    if not has_import:
        result.errors.append(
            "Missing vibetrading import. Strategy must import from vibetrading "
            "(e.g., `from vibetrading import vibe, get_perp_price, ...`)."
        )


def _check_vibe_decorator(code: str, tree: ast.Module, result: StrategyValidationResult):
    """Check that exactly one function has the @vibe decorator."""
    vibe_functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                decorator_name = _get_decorator_name(decorator)
                if decorator_name in ("vibe", "vibetrading.vibe"):
                    vibe_functions.append(node.name)

    if len(vibe_functions) == 0:
        result.errors.append(
            "No function decorated with `@vibe`. "
            "Exactly one strategy function must have the @vibe decorator."
        )
    elif len(vibe_functions) > 1:
        result.errors.append(
            f"Multiple @vibe decorated functions found: {vibe_functions}. "
            f"Only ONE function may have @vibe. Use helper functions for organization."
        )


def _get_decorator_name(node: ast.expr) -> str:
    """Extract decorator name from AST node."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
    elif isinstance(node, ast.Call):
        return _get_decorator_name(node.func)
    return ""


def _check_leverage(code: str, result: StrategyValidationResult):
    """Check that set_leverage is called before long/short."""
    has_long_or_short = bool(re.search(r"\blong\s*\(", code))
    has_short = bool(re.search(r"\bshort\s*\(", code))
    has_set_leverage = bool(re.search(r"\bset_leverage\s*\(", code))

    if (has_long_or_short or has_short) and not has_set_leverage:
        result.errors.append(
            "Strategy uses `long()` or `short()` but never calls `set_leverage()`. "
            "Must call `set_leverage(asset, leverage)` before futures trading."
        )


def _check_data_validation(code: str, result: StrategyValidationResult):
    """Check for DataFrame length validation before index access."""
    uses_ohlcv = bool(re.search(r"\bget_(futures|spot)_ohlcv\s*\(", code))
    has_len_check = bool(
        re.search(r"len\s*\(.+\)\s*(<|<=|>=|>)\s*\d+", code)
        or re.search(r"if\s+len\s*\(", code)
        or re.search(r"\.shape\[0\]\s*(<|<=|>=|>)", code)
    )

    if uses_ohlcv and not has_len_check:
        result.warnings.append(
            "Strategy fetches OHLCV data but does not check DataFrame length. "
            "Always validate `len(df) >= N` before accessing rolling/iloc to avoid "
            "IndexError on insufficient data."
        )


def _check_position_sizing(code: str, result: StrategyValidationResult):
    """Check for dynamic balance-based position sizing."""
    has_trading = bool(re.search(r"\b(long|short|buy|sell)\s*\(", code))
    if not has_trading:
        return

    has_balance_check = bool(
        re.search(r"\bget_perp_summary\s*\(", code)
        or re.search(r"\bget_spot_summary\s*\(", code)
        or re.search(r"\bmy_(futures|spot)_balance\s*\(", code)
    )

    if not has_balance_check:
        result.warnings.append(
            "Strategy places orders but does not query account balance. "
            "Use `get_perp_summary()` or `get_spot_summary()` for dynamic position sizing."
        )


def _check_risk_management(code: str, result: StrategyValidationResult):
    """Check for take-profit and stop-loss logic."""
    has_futures_trading = bool(re.search(r"\b(long|short)\s*\(", code))
    if not has_futures_trading:
        return

    has_reduce = bool(re.search(r"\breduce_position\s*\(", code))
    has_tp_sl_keywords = bool(
        re.search(r"(tp|take.?profit|sl|stop.?loss|TP_PCT|SL_PCT)", code, re.IGNORECASE)
    )

    if not has_reduce:
        result.warnings.append(
            "Strategy opens futures positions but never calls `reduce_position()`. "
            "Every strategy must implement take-profit and stop-loss logic."
        )
    elif not has_tp_sl_keywords:
        result.warnings.append(
            "Strategy uses `reduce_position()` but has no clear TP/SL logic. "
            "Consider adding explicit take-profit and stop-loss thresholds."
        )


def _check_nan_handling(code: str, result: StrategyValidationResult):
    """Check for NaN handling when using price functions."""
    uses_price = bool(re.search(r"\bget_(perp|spot)_price\s*\(", code))
    has_nan_check = bool(
        re.search(r"math\.isnan\s*\(", code)
        or re.search(r"pd\.isna\s*\(", code)
        or re.search(r"np\.isnan\s*\(", code)
    )

    if uses_price and not has_nan_check:
        result.warnings.append(
            "Strategy calls price functions but does not check for NaN. "
            "Use `if math.isnan(price): return` to handle unavailable prices."
        )


def _check_hardcoded_balance(code: str, result: StrategyValidationResult):
    """Check for hardcoded balance amounts."""
    # Look for suspicious patterns like `initial_capital = 10000` or `balance = 5000`
    has_hardcoded = bool(
        re.search(
            r"\b(initial_capital|starting_balance|total_capital|initial_balance)\s*=\s*\d+",
            code,
        )
    )

    if has_hardcoded:
        result.warnings.append(
            "Possible hardcoded balance amount detected. Never hardcode capital. "
            "Use `get_perp_summary()` or `get_spot_summary()` to query balance dynamically."
        )


def _check_deprecated_functions(code: str, result: StrategyValidationResult):
    """Check for use of deprecated API functions."""
    deprecated = {
        r"\bmy_spot_balance\s*\(": "Use `get_spot_summary()` instead of `my_spot_balance()`",
        r"\bmy_futures_balance\s*\(": "Use `get_perp_summary()` instead of `my_futures_balance()`",
        r"\bget_futures_position\s*\(": "Use `get_perp_position(asset)` instead of `get_futures_position()`",
        r"\bcancel_order\s*\(": "Use `cancel_perp_orders(asset, ids)` or `cancel_spot_orders(asset, ids)` instead of `cancel_order()`",
    }

    for pattern, message in deprecated.items():
        if re.search(pattern, code):
            result.warnings.append(f"Deprecated function usage: {message}")
