"""
LiveRunner - Run strategy code against a live exchange sandbox.

Provides the runtime for executing user strategies with a real exchange sandbox,
including the mock vibetrading module injection, callback registration, and
a periodic execution loop.
"""

import importlib
import importlib.util
import sys
import io
import contextlib
import traceback
import asyncio
import logging
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Any

import numpy as np
import pandas as pd

from .sandbox_base import VibeSandboxBase, SUPPORTED_INTERVALS
from .error_handler import StrategyErrorHandler
from .._utils.logging import log_runtime_error

logger = logging.getLogger(__name__)


class LiveRunner:
    """
    Runs a strategy against a live exchange sandbox.

    Usage::

        from vibetrading.exchanges import create_sandbox
        from vibetrading.core.live_runner import LiveRunner

        sandbox = create_sandbox("hyperliquid", api_key=..., api_secret=...)
        runner = LiveRunner(sandbox)
        runner.load_strategy(strategy_code)
        await runner.start()
    """

    def __init__(
        self,
        sandbox: VibeSandboxBase,
        interval: str = "1m",
    ):
        self.sandbox = sandbox
        self.interval = interval
        self.interval_seconds = self._parse_interval(interval)
        self.registered_strategy_callbacks: List[Callable] = []
        self.strategy_code: Optional[str] = None
        self.should_stop = False

        # Module injection tracking
        self._orig_in_sys = False
        self._orig_mod = None
        self._mock_mod = None

    @staticmethod
    def _parse_interval(interval: str) -> int:
        interval = interval.lower()
        if interval == "1s":
            return 1
        if interval.endswith("m"):
            return int(interval[:-1]) * 60
        if interval.endswith("h"):
            return int(interval[:-1]) * 3600
        if interval.endswith("d"):
            return int(interval[:-1]) * 86400
        if interval == "1w":
            return 604800
        return 60

    def load_strategy(self, strategy_code: str):
        """
        Load and register the strategy code.

        Creates a mock ``vibetrading`` module, injects all sandbox functions,
        and executes the strategy code via ``exec()``.
        """
        self.strategy_code = strategy_code
        self.registered_strategy_callbacks.clear()

        self._orig_in_sys = 'vibetrading' in sys.modules
        self._orig_mod = sys.modules.get('vibetrading')

        spec = importlib.util.spec_from_loader('vibetrading', loader=None)
        if spec is None:
            raise ImportError("Cannot create vibetrading module spec")
        self._mock_mod = importlib.util.module_from_spec(spec)

        error_handler = StrategyErrorHandler(self.sandbox, strategy_code)

        runner = self

        def mock_vibe(interval_or_func=None, **kwargs):
            if callable(interval_or_func):
                func = interval_or_func
                runner.registered_strategy_callbacks.append(error_handler.wrap_strategy(func))
                return func
            else:
                ival = interval_or_func or kwargs.get("interval", "1m")
                runner.interval = ival
                runner.interval_seconds = LiveRunner._parse_interval(ival)
                if hasattr(runner.sandbox, 'set_run_interval'):
                    runner.sandbox.set_run_interval(ival)
                def dec(func):
                    runner.registered_strategy_callbacks.append(error_handler.wrap_strategy(func))
                    return func
                return dec

        funcs = {
            'my_spot_balance': self.sandbox.my_spot_balance,
            'my_futures_balance': self.sandbox.my_futures_balance,
            'buy': self.sandbox.buy, 'sell': self.sandbox.sell,
            'long': self.sandbox.long, 'short': self.sandbox.short,
            'reduce_position': self.sandbox.reduce_position,
            'set_leverage': self.sandbox.set_leverage,
            'get_futures_position': self.sandbox.get_futures_position,
            'get_price': self.sandbox.get_price,
            'get_spot_ohlcv': self.sandbox.get_spot_ohlcv,
            'get_futures_ohlcv': self.sandbox.get_futures_ohlcv,
            'get_funding_rate': self.sandbox.get_funding_rate,
            'get_funding_rate_history': self.sandbox.get_funding_rate_history,
            'get_open_interest': self.sandbox.get_open_interest,
            'get_open_interest_history': self.sandbox.get_open_interest_history,
            'get_spot_price': self.sandbox.get_spot_price,
            'get_perp_price': self.sandbox.get_perp_price,
            'get_spot_summary': self.sandbox.get_spot_summary,
            'get_perp_summary': self.sandbox.get_perp_summary,
            'cancel_spot_orders': self.sandbox.cancel_spot_orders,
            'cancel_perp_orders': self.sandbox.cancel_perp_orders,
            'get_perp_position': self.sandbox.get_perp_position,
            'get_current_time': self.sandbox.get_current_time,
            'get_supported_assets': self.sandbox.get_supported_assets,
            'get_perp_open_orders': self.sandbox.get_perp_open_orders,
            'get_spot_open_orders': self.sandbox.get_spot_open_orders,
            'cancel_order': self.sandbox.cancel_order,
            'vibe': mock_vibe,
        }

        for name, obj in funcs.items():
            setattr(self._mock_mod, name, obj)
        sys.modules['vibetrading'] = self._mock_mod

        exec_globals = {'pd': pd, 'np': np, 'ta': None}
        try:
            import ta
            exec_globals['ta'] = ta
        except ImportError:
            pass
        exec_globals.update(funcs)

        exec(strategy_code, exec_globals)

        if not self.registered_strategy_callbacks:
            raise ValueError("No strategy functions registered. Use @vibe decorator.")

        logger.info("Strategy loaded: %d callback(s), interval=%s",
                     len(self.registered_strategy_callbacks), self.interval)

    def cleanup(self):
        """Restore the original vibetrading module."""
        if self._mock_mod and 'vibetrading' in sys.modules and sys.modules['vibetrading'] is self._mock_mod:
            if self._orig_in_sys and self._orig_mod is not None:
                sys.modules['vibetrading'] = self._orig_mod
            elif not self._orig_in_sys:
                del sys.modules['vibetrading']
        self._mock_mod = None

    def run_callbacks_once(self):
        """Execute all registered callbacks once (synchronous)."""
        for cb in self.registered_strategy_callbacks:
            try:
                captured = io.StringIO()
                with contextlib.redirect_stdout(captured):
                    cb()
                output = captured.getvalue().strip()
                if output:
                    logger.info("Strategy output: %s", output)
            except Exception as e:
                tb = traceback.format_exc()
                log_runtime_error(type(e).__name__, str(e), tb, getattr(cb, '__name__', 'callback'))
                raise

    async def start(self):
        """Start the periodic execution loop (async)."""
        logger.info("LiveRunner starting (interval=%ss)", self.interval_seconds)
        try:
            while not self.should_stop:
                try:
                    self.run_callbacks_once()
                except Exception as e:
                    logger.error("Strategy error: %s", e)
                    # Continue running, don't break for single errors
                for _ in range(self.interval_seconds):
                    if self.should_stop:
                        break
                    await asyncio.sleep(1)
        finally:
            self.cleanup()
        logger.info("LiveRunner stopped")

    def stop(self):
        """Signal the runner to stop after the current iteration."""
        self.should_stop = True
