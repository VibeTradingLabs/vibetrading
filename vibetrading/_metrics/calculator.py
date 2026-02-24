"""
MetricsCalculator - Consolidated metrics calculation for backtesting.

Provides a single source of truth for all performance metrics calculation
including return, Sharpe ratio, max drawdown, win rate, funding revenue,
and trade duration.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np


@dataclass
class MetricsContext:
    """Context for metrics calculation with reusable data."""
    initial_value: float
    current_value: float
    results: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    funding_payments: List[Dict[str, Any]]
    interval: str
    use_all_data: bool

    all_time_peak: Optional[float] = None
    max_drawdown_observed: Optional[float] = None
    win_rate_tracker: Optional[Dict[str, Any]] = None
    sharpe_tracker: Optional[Dict[str, Any]] = None
    total_tx_fees: float = 0.0


class MetricsCalculator:
    """
    Consolidated metrics calculator for backtesting.

    Handles all performance metrics with consistent methodology
    for both live (streaming) and final (complete) calculations.
    """

    def __init__(self, backtest):
        """
        Args:
            backtest: Reference to BacktestEngine instance for accessing state
        """
        self.backtest = backtest

    def calculate(self, use_all_data: bool = True) -> Dict[str, float]:
        """Calculate all metrics from current backtest state."""
        ctx = self._build_context(use_all_data)

        if not ctx.results or len(ctx.results) < 2:
            return self._empty_metrics(ctx)

        return {
            'total_return': self._calculate_return(ctx),
            'max_drawdown': self._calculate_drawdown(ctx),
            'sharpe_ratio': self._calculate_sharpe(ctx),
            'win_rate': self._calculate_win_rate(ctx),
            'total_trades': len(ctx.trades),
            'funding_revenue': self._calculate_funding_revenue(ctx),
            'total_tx_fees': ctx.total_tx_fees,
            'average_trade_duration_hours': self._calculate_avg_trade_duration(ctx),
        }

    def _build_context(self, use_all_data: bool) -> MetricsContext:
        """Build metrics context from current backtest state."""
        bt = self.backtest

        initial_value = (bt._initial_total_value
                        if bt._initial_total_value is not None
                        else bt.sandbox.INITIAL_USDC_BALANCE)

        current_value = bt._calculate_total_value()

        return MetricsContext(
            initial_value=initial_value,
            current_value=current_value,
            results=bt.results,
            trades=bt.sandbox.trades if hasattr(bt.sandbox, 'trades') else [],
            funding_payments=bt.sandbox.funding_payments if hasattr(bt.sandbox, 'funding_payments') else [],
            interval=bt.interval,
            use_all_data=use_all_data,
            all_time_peak=bt._all_time_peak,
            max_drawdown_observed=bt._max_drawdown_observed if hasattr(bt, '_max_drawdown_observed') else None,
            win_rate_tracker=bt._win_rate_tracker if hasattr(bt, '_win_rate_tracker') else None,
            sharpe_tracker=bt._sharpe_tracker if hasattr(bt, '_sharpe_tracker') else None,
            total_tx_fees=bt.sandbox.total_tx_fees if hasattr(bt.sandbox, 'total_tx_fees') else 0.0,
        )

    def _empty_metrics(self, ctx: MetricsContext) -> Dict[str, float]:
        """Return empty metrics when insufficient data."""
        return {
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'total_trades': len(ctx.trades),
            'sharpe_ratio': 0.0,
            'total_return': 0.0,
            'funding_revenue': 0.0,
            'total_tx_fees': 0.0,
            'average_trade_duration_hours': 0.0,
        }

    def _calculate_return(self, ctx: MetricsContext) -> float:
        """Calculate total return (decimal 0-1 scale, not percentage)."""
        if ctx.initial_value <= 0:
            return 0.0
        return (ctx.current_value - ctx.initial_value) / ctx.initial_value

    def _calculate_drawdown(self, ctx: MetricsContext) -> float:
        """Calculate maximum drawdown."""
        if ctx.use_all_data:
            return self._calculate_drawdown_full(ctx)
        else:
            return self._calculate_drawdown_live(ctx)

    def _calculate_drawdown_full(self, ctx: MetricsContext) -> float:
        """Calculate max drawdown from all portfolio values."""
        values = [result["total_value"] for result in ctx.results]
        if not values:
            return 0.0

        peak = values[0]
        max_drawdown = 0.0
        for value in values:
            if value > peak:
                peak = value
            drawdown = (value - peak) / peak if peak > 0 else 0.0
            if drawdown < max_drawdown:
                max_drawdown = drawdown

        return min(0.0, max_drawdown)

    def _calculate_drawdown_live(self, ctx: MetricsContext) -> float:
        """Calculate max drawdown using tracked all-time peak."""
        if ctx.max_drawdown_observed is not None:
            return ctx.max_drawdown_observed

        if ctx.all_time_peak is None:
            recent_results = ctx.results[-100:] if len(ctx.results) > 100 else ctx.results
            values = [result["total_value"] for result in recent_results]
            return self._calculate_drawdown_from_values(values)

        if ctx.all_time_peak <= 0:
            return 0.0

        current_value = ctx.results[-1]["total_value"] if ctx.results else ctx.current_value
        drawdown = (current_value - ctx.all_time_peak) / ctx.all_time_peak
        return min(0.0, drawdown)

    def _calculate_drawdown_from_values(self, values: List[float]) -> float:
        """Helper to calculate drawdown from a list of values."""
        if not values:
            return 0.0
        peak = values[0]
        max_drawdown = 0.0
        for value in values:
            if value > peak:
                peak = value
            drawdown = (value - peak) / peak if peak > 0 else 0.0
            if drawdown < max_drawdown:
                max_drawdown = drawdown
        return min(0.0, max_drawdown)

    def _calculate_sharpe(self, ctx: MetricsContext, risk_free_rate: float = 0.0) -> float:
        """Calculate annualized Sharpe ratio using Welford's algorithm."""
        if ctx.sharpe_tracker is None or ctx.sharpe_tracker['return_count'] < 2:
            return 0.0

        tracker = ctx.sharpe_tracker
        n = tracker['return_count']
        mean_return = tracker.get('mean', 0.0)
        m2 = tracker.get('m2', 0.0)
        variance = m2 / (n - 1)

        if variance <= 0:
            return 0.0

        std_dev = variance ** 0.5
        periods_per_year = self._get_periods_per_year(ctx.interval)
        excess_return = mean_return - (risk_free_rate / periods_per_year)
        return (excess_return / std_dev) * (periods_per_year ** 0.5)

    def _get_periods_per_year(self, interval: str) -> float:
        """Calculate number of periods per year for given interval."""
        if interval.endswith('m'):
            minutes = int(interval[:-1])
            return 365 * 24 * 60 / minutes
        elif interval.endswith('h'):
            hours = int(interval[:-1])
            return 365 * 24 / hours
        elif interval.endswith('d'):
            days = int(interval[:-1])
            return 365 / days
        else:
            return 252.0

    def _calculate_win_rate(self, ctx: MetricsContext) -> float:
        """Calculate win rate from trade outcomes."""
        if not ctx.trades:
            return 0.0
        if ctx.use_all_data:
            return self._calculate_win_rate_full(ctx)
        else:
            return self._calculate_win_rate_live(ctx)

    def _calculate_win_rate_full(self, ctx: MetricsContext) -> float:
        """Calculate win rate from all trades."""
        winning_trades = 0
        losing_trades = 0

        for trade in ctx.trades:
            pnl = None
            if trade.get('action') in ['sell', 'close_position'] and 'pnl' in trade:
                pnl = trade.get('pnl', 0.0)
            elif 'realized_pnl' in trade:
                pnl = trade.get('realized_pnl', 0.0)

            if pnl is not None:
                if pnl > 0:
                    winning_trades += 1
                elif pnl < 0:
                    losing_trades += 1

        total_closed_trades = winning_trades + losing_trades
        if total_closed_trades > 0:
            return winning_trades / total_closed_trades

        return self._estimate_win_rate_from_returns(ctx)

    def _calculate_win_rate_live(self, ctx: MetricsContext) -> float:
        """Calculate win rate using incremental tracker."""
        self.backtest._update_win_rate_tracker_incremental()
        tracker = self.backtest._win_rate_tracker if hasattr(self.backtest, '_win_rate_tracker') else ctx.win_rate_tracker

        if tracker is None:
            return self._estimate_win_rate_from_returns(ctx)

        total_closed_trades = tracker['total_closed_trades']
        winning_trades = tracker['winning_trades']

        if total_closed_trades > 0:
            return winning_trades / total_closed_trades

        return self._estimate_win_rate_from_returns(ctx)

    def _estimate_win_rate_from_returns(self, ctx: MetricsContext) -> float:
        """Estimate win rate from portfolio returns as a fallback heuristic."""
        if ctx.current_value > ctx.initial_value:
            return_ratio = (ctx.current_value - ctx.initial_value) / ctx.initial_value
            return min(0.70, 0.35 + (return_ratio * 0.8))
        else:
            return_ratio = (ctx.initial_value - ctx.current_value) / ctx.initial_value
            return max(0.15, 0.35 - (return_ratio * 0.8))

    def _calculate_funding_revenue(self, ctx: MetricsContext) -> float:
        """Calculate total funding revenue from funding payments."""
        if not ctx.funding_payments:
            return 0.0
        return -sum(payment.get('funding_payment', 0.0) for payment in ctx.funding_payments)

    def _calculate_avg_trade_duration(self, ctx: MetricsContext) -> float:
        """Calculate average trade duration in hours."""
        if len(ctx.trades) < 2:
            return 0.0
        if ctx.use_all_data:
            return self._calculate_avg_duration_accurate(ctx.trades)
        else:
            return self._calculate_avg_duration_approximate(ctx.trades)

    def _calculate_avg_duration_accurate(self, trades: List[Dict]) -> float:
        """Calculate accurate average duration by grouping trades by asset."""
        trade_durations = []
        trades_by_asset = {}

        for trade in trades:
            asset = trade.get('asset', '')
            if asset not in trades_by_asset:
                trades_by_asset[asset] = []
            trades_by_asset[asset].append(trade)

        for asset, asset_trades in trades_by_asset.items():
            sorted_trades = sorted(asset_trades, key=lambda t: t.get('time', datetime.min))
            for i in range(1, len(sorted_trades)):
                prev_time = sorted_trades[i-1].get('time')
                curr_time = sorted_trades[i].get('time')
                if prev_time and curr_time and isinstance(prev_time, datetime) and isinstance(curr_time, datetime):
                    duration = (curr_time - prev_time).total_seconds() / 3600.0
                    if duration > 0:
                        trade_durations.append(duration)

        return sum(trade_durations) / len(trade_durations) if trade_durations else 0.0

    def _calculate_avg_duration_approximate(self, trades: List[Dict]) -> float:
        """Calculate approximate average duration for performance."""
        if len(trades) < 2:
            return 0.0
        first_time = trades[0].get('time')
        last_time = trades[-1].get('time')
        if first_time and last_time and isinstance(first_time, datetime) and isinstance(last_time, datetime):
            total_hours = (last_time - first_time).total_seconds() / 3600.0
            if total_hours > 0:
                return total_hours / len(trades)
        return 0.0
