"""
StrategyErrorHandler - Consolidated error handling for strategy execution.

Wraps strategy functions with comprehensive error logging and context extraction,
providing detailed traceback information for debugging and auto-regeneration.
"""

import traceback
from typing import Callable, Optional, Dict
from functools import wraps

from ..utils.logging import log_runtime_error


class StrategyErrorHandler:
    """
    Handles strategy execution errors with detailed context extraction.

    Wraps strategy functions to catch exceptions and provide:
    - Detailed error information
    - Strategy code context extraction
    - Structured logging for regeneration systems
    """

    def __init__(self, sandbox, strategy_code: str):
        """
        Args:
            sandbox: The trading sandbox instance (for accessing current_time)
            strategy_code: The complete strategy code string for context extraction
        """
        self.sandbox = sandbox
        self.strategy_code = strategy_code

    def wrap_strategy(self, func: Callable) -> Callable:
        """Wrap strategy function with error handling."""
        @wraps(func)
        def wrapper():
            try:
                print(f"Executing strategy function: {func.__name__}")
                result = func()
                print(f"Strategy function completed successfully")
                return result
            except Exception as e:
                self._handle_error(e, func.__name__)
                raise
        return wrapper

    def _handle_error(self, error: Exception, func_name: str):
        """Handle strategy execution error with detailed logging."""
        error_type = type(error).__name__
        error_message = str(error)
        error_traceback = traceback.format_exc()

        error_context = self._extract_error_context(error)

        print(f"Strategy Error in {func_name}: {error_type} - {error_message}")
        print(f"  Function: {func_name}")
        print(f"  Error Type: {error_type}")
        print(f"  Error Message: {error_message}")
        print(f"  Time: {self.sandbox.current_time}")

        if error_context:
            self._print_error_context(error_context)

        print(f"  Full traceback:")
        print(error_traceback)

        log_runtime_error(
            error_type=error_type,
            error_message=error_message,
            error_traceback=error_traceback,
            strategy_function=func_name,
            timestamp=self.sandbox.current_time.isoformat() if self.sandbox.current_time else None,
            strategy_code_context=error_context
        )

    def _extract_error_context(self, error: Exception) -> Optional[Dict]:
        """Extract strategy code context from error traceback."""
        try:
            tb = error.__traceback__
            strategy_frames = []

            while tb is not None:
                frame = tb.tb_frame
                filename = frame.f_code.co_filename
                line_no = tb.tb_lineno

                if filename == '<string>':
                    if self.strategy_code:
                        strategy_lines = self.strategy_code.split('\n')
                        if 1 <= line_no <= len(strategy_lines):
                            error_line = strategy_lines[line_no - 1].strip()
                            strategy_frames.append({
                                'line_no': line_no,
                                'code': error_line,
                                'function': frame.f_code.co_name
                            })
                tb = tb.tb_next

            if strategy_frames:
                return {
                    'error_frames': strategy_frames,
                    'primary_error_line': strategy_frames[-1]['line_no'],
                    'primary_error_code': strategy_frames[-1]['code']
                }
            return None

        except Exception:
            return None

    def _print_error_context(self, context: Dict):
        """Print formatted error context showing code lines involved."""
        print(f"  Strategy Code Error Details:")
        for frame in context['error_frames']:
            print(f"    Line {frame['line_no']} in {frame['function']}: {frame['code']}")
        print(f"  Primary error at line {context['primary_error_line']}: {context['primary_error_code']}")
