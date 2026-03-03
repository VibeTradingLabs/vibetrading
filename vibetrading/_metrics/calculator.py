"""
MetricsCalculator - Comprehensive performance metrics for backtesting.

Provides a single source of truth for all performance metrics calculation
including return, risk-adjusted returns (Sharpe, Sortino, Calmar), drawdown
analysis, trade statistics, and funding revenue.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class MetricsContext:
    """Context for metrics calculation with reusable data."""

    initial_value: float
    current_value: float
    results: list[dict[str, Any]]
    trades: list[dict[str, Any]]
    funding_payments: list[dict[str, Any]]
    interval: str
    use_all_data: bool

    all_time_peak: float | None = None
    max_drawdown_observed: float | None = None
    win_rate_tracker: dict[str, Any] | None = None
    sharpe_tracker: dict[str, Any] | None = None
    total_tx_fees: float = 0.0


@dataclass
class TradeStats:
    """Detailed trade-level statistics."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_trade_pnl: float = 0.0
    total_pnl: float = 0.0
    expectancy: float = 0.0


class MetricsCalculator:
    """
    Comprehensive metrics calculator for backtesting.

    Handles all performance metrics with consistent methodology
    for both live (streaming) and final (complete) calculations.

    Metrics include:
    - Total return and CAGR
    - Sharpe, Sortino, and Calmar ratios
    - Maximum drawdown and drawdown duration
    - Win rate, profit factor, expectancy
    - Consecutive win/loss streaks
    - Funding revenue and transaction fees
    """

    def __init__(self, backtest):
        """
        Args:
            backtest: Reference to BacktestEngine instance for accessing state
        """
        self.backtest = backtest

    def calculate(self, use_all_data: bool = True) -> dict[str, float]:
        """Calculate all metrics from current backtest state."""
        ctx = self._build_context(use_all_data)

        if not ctx.results or len(ctx.results) < 2:
            return self._empty_metrics(ctx)

        trade_stats = self._calculate_trade_stats(ctx)
        returns = self._extract_returns(ctx)

        return {
            "total_return": self._calculate_return(ctx),
            "cagr": self._calculate_cagr(ctx),
            "max_drawdown": self._calculate_drawdown(ctx),
            "max_drawdown_duration_hours": self._calculate_drawdown_duration(ctx),
            "sharpe_ratio": self._calculate_sharpe(ctx),
            "sortino_ratio": self._calculate_sortino(ctx, returns),
            "calmar_ratio": self._calculate_calmar(ctx),
            "win_rate": trade_stats.win_rate,
            "profit_factor": trade_stats.profit_factor,
            "expectancy": trade_stats.expectancy,
            "avg_win": trade_stats.avg_win,
            "avg_loss": trade_stats.avg_loss,
            "largest_win": trade_stats.largest_win,
            "largest_loss": trade_stats.largest_loss,
            "max_consecutive_wins": trade_stats.max_consecutive_wins,
            "max_consecutive_losses": trade_stats.max_consecutive_losses,
            "total_trades": trade_stats.total_trades,
            "winning_trades": trade_stats.winning_trades,
            "losing_trades": trade_stats.losing_trades,
            "avg_trade_pnl": trade_stats.avg_trade_pnl,
            "total_pnl": trade_stats.total_pnl,
            "funding_revenue": self._calculate_funding_revenue(ctx),
            "total_tx_fees": ctx.total_tx_fees,
            "average_trade_duration_hours": self._calculate_avg_trade_duration(ctx),
        }

    def _build_context(self, use_all_data: bool) -> MetricsContext:
        """Build metrics context from current backtest state."""
        bt = self.backtest

        initial_value = (
            bt._initial_total_value if bt._initial_total_value is not None else bt.sandbox.INITIAL_USDC_BALANCE
        )

        current_value = bt._calculate_total_value()

        return MetricsContext(
            initial_value=initial_value,
            current_value=current_value,
            results=bt.results,
            trades=bt.sandbox.trades if hasattr(bt.sandbox, "trades") else [],
            funding_payments=bt.sandbox.funding_payments if hasattr(bt.sandbox, "funding_payments") else [],
            interval=bt.interval,
            use_all_data=use_all_data,
            all_time_peak=bt._all_time_peak,
            max_drawdown_observed=bt._max_drawdown_observed if hasattr(bt, "_max_drawdown_observed") else None,
            win_rate_tracker=bt._win_rate_tracker if hasattr(bt, "_win_rate_tracker") else None,
            sharpe_tracker=bt._sharpe_tracker if hasattr(bt, "_sharpe_tracker") else None,
            total_tx_fees=bt.sandbox.total_tx_fees if hasattr(bt.sandbox, "total_tx_fees") else 0.0,
        )

    def _empty_metrics(self, ctx: MetricsContext) -> dict[str, float]:
        """Return empty metrics when insufficient data."""
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_duration_hours": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "total_trades": len(ctx.trades),
            "winning_trades": 0,
            "losing_trades": 0,
            "avg_trade_pnl": 0.0,
            "total_pnl": 0.0,
            "funding_revenue": 0.0,
            "total_tx_fees": 0.0,
            "average_trade_duration_hours": 0.0,
        }

    # ── Return metrics ─────────────────────────────────────────────────

    def _calculate_return(self, ctx: MetricsContext) -> float:
        """Calculate total return (decimal 0-1 scale, not percentage)."""
        if ctx.initial_value <= 0:
            return 0.0
        return (ctx.current_value - ctx.initial_value) / ctx.initial_value

    def _calculate_cagr(self, ctx: MetricsContext) -> float:
        """Calculate Compound Annual Growth Rate."""
        if ctx.initial_value <= 0 or ctx.current_value <= 0:
            return 0.0
        if len(ctx.results) < 2:
            return 0.0

        # Estimate duration from number of periods
        n_periods = len(ctx.results)
        periods_per_year = self._get_periods_per_year(ctx.interval)
        years = n_periods / periods_per_year

        if years <= 0:
            return 0.0

        return (ctx.current_value / ctx.initial_value) ** (1 / years) - 1

    # ── Risk metrics ───────────────────────────────────────────────────

    def _extract_returns(self, ctx: MetricsContext) -> list[float]:
        """Extract period-over-period returns from results."""
        values = [r["total_value"] for r in ctx.results]
        if len(values) < 2:
            return []
        returns = []
        for i in range(1, len(values)):
            if values[i - 1] > 0:
                returns.append((values[i] - values[i - 1]) / values[i - 1])
        return returns

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
        return self._calculate_drawdown_from_values(values)

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

    def _calculate_drawdown_from_values(self, values: list[float]) -> float:
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

    def _calculate_drawdown_duration(self, ctx: MetricsContext) -> float:
        """Calculate the longest drawdown duration in hours."""
        values = [r["total_value"] for r in ctx.results]
        if len(values) < 2:
            return 0.0

        interval_hours = self._get_interval_hours(ctx.interval)
        peak = values[0]
        max_duration = 0
        current_duration = 0

        for value in values:
            if value >= peak:
                peak = value
                current_duration = 0
            else:
                current_duration += 1
                if current_duration > max_duration:
                    max_duration = current_duration

        return max_duration * interval_hours

    def _calculate_sharpe(self, ctx: MetricsContext, risk_free_rate: float = 0.0) -> float:
        """Calculate annualized Sharpe ratio using Welford's algorithm."""
        if ctx.sharpe_tracker is None or ctx.sharpe_tracker["return_count"] < 2:
            return 0.0

        tracker = ctx.sharpe_tracker
        n = tracker["return_count"]
        mean_return = tracker.get("mean", 0.0)
        m2 = tracker.get("m2", 0.0)
        variance = m2 / (n - 1)

        if variance <= 0:
            return 0.0

        std_dev = variance**0.5
        periods_per_year = self._get_periods_per_year(ctx.interval)
        excess_return = mean_return - (risk_free_rate / periods_per_year)
        return (excess_return / std_dev) * (periods_per_year**0.5)

    def _calculate_sortino(self, ctx: MetricsContext, returns: list[float], risk_free_rate: float = 0.0) -> float:
        """Calculate annualized Sortino ratio (penalizes only downside volatility)."""
        if len(returns) < 2:
            return 0.0

        periods_per_year = self._get_periods_per_year(ctx.interval)
        rf_per_period = risk_free_rate / periods_per_year

        excess_returns = [r - rf_per_period for r in returns]
        mean_excess = sum(excess_returns) / len(excess_returns)

        # Downside deviation: std of negative returns only
        downside_returns = [r for r in excess_returns if r < 0]
        if not downside_returns:
            # No downside — infinite Sortino; cap at a reasonable value
            return 100.0 if mean_excess > 0 else 0.0

        downside_sq = sum(r**2 for r in downside_returns) / len(returns)
        downside_dev = math.sqrt(downside_sq)

        if downside_dev <= 0:
            return 0.0

        return (mean_excess / downside_dev) * math.sqrt(periods_per_year)

    def _calculate_calmar(self, ctx: MetricsContext) -> float:
        """Calculate Calmar ratio (CAGR / max drawdown)."""
        cagr = self._calculate_cagr(ctx)
        max_dd = abs(self._calculate_drawdown(ctx))

        if max_dd <= 0:
            return 100.0 if cagr > 0 else 0.0

        return cagr / max_dd

    # ── Trade statistics ───────────────────────────────────────────────

    def _calculate_trade_stats(self, ctx: MetricsContext) -> TradeStats:
        """Calculate comprehensive trade-level statistics."""
        stats = TradeStats()

        pnls = self._extract_trade_pnls(ctx.trades)
        if not pnls:
            # Fall back to win rate estimation if no PnL data
            stats.total_trades = len(ctx.trades)
            stats.win_rate = self._estimate_win_rate_from_returns(ctx) if ctx.trades else 0.0
            return stats

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        breakeven = [p for p in pnls if p == 0]

        stats.total_trades = len(ctx.trades)
        stats.winning_trades = len(wins)
        stats.losing_trades = len(losses)
        stats.breakeven_trades = len(breakeven)

        total_closed = stats.winning_trades + stats.losing_trades
        stats.win_rate = stats.winning_trades / total_closed if total_closed > 0 else 0.0

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        stats.profit_factor = gross_profit / gross_loss if gross_loss > 0 else (100.0 if gross_profit > 0 else 0.0)

        stats.avg_win = gross_profit / len(wins) if wins else 0.0
        stats.avg_loss = -gross_loss / len(losses) if losses else 0.0
        stats.largest_win = max(wins) if wins else 0.0
        stats.largest_loss = min(losses) if losses else 0.0

        stats.total_pnl = sum(pnls)
        stats.avg_trade_pnl = stats.total_pnl / len(pnls) if pnls else 0.0

        # Expectancy: avg_win * win_rate - avg_loss * loss_rate
        if total_closed > 0:
            loss_rate = stats.losing_trades / total_closed
            stats.expectancy = (stats.avg_win * stats.win_rate) + (stats.avg_loss * loss_rate)

        # Consecutive wins/losses
        stats.max_consecutive_wins, stats.max_consecutive_losses = self._calculate_streaks(pnls)

        return stats

    def _extract_trade_pnls(self, trades: list[dict]) -> list[float]:
        """Extract PnL values from trade records."""
        pnls = []
        for trade in trades:
            pnl = None
            if trade.get("action") in ["sell", "close_position"] and "pnl" in trade:
                pnl = trade.get("pnl", 0.0)
            elif "realized_pnl" in trade:
                pnl = trade.get("realized_pnl", 0.0)
            if pnl is not None:
                pnls.append(pnl)
        return pnls

    def _calculate_streaks(self, pnls: list[float]) -> tuple[int, int]:
        """Calculate max consecutive wins and losses."""
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for pnl in pnls:
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                if current_wins > max_wins:
                    max_wins = current_wins
            elif pnl < 0:
                current_losses += 1
                current_wins = 0
                if current_losses > max_losses:
                    max_losses = current_losses
            else:
                current_wins = 0
                current_losses = 0

        return max_wins, max_losses

    # ── Legacy compatibility ───────────────────────────────────────────

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
        pnls = self._extract_trade_pnls(ctx.trades)
        if not pnls:
            return self._estimate_win_rate_from_returns(ctx)

        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p < 0)
        total = wins + losses
        return wins / total if total > 0 else self._estimate_win_rate_from_returns(ctx)

    def _calculate_win_rate_live(self, ctx: MetricsContext) -> float:
        """Calculate win rate using incremental tracker."""
        self.backtest._update_win_rate_tracker_incremental()
        tracker = (
            self.backtest._win_rate_tracker if hasattr(self.backtest, "_win_rate_tracker") else ctx.win_rate_tracker
        )

        if tracker is None:
            return self._estimate_win_rate_from_returns(ctx)

        total_closed_trades = tracker["total_closed_trades"]
        winning_trades = tracker["winning_trades"]

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

    # ── Funding & fees ─────────────────────────────────────────────────

    def _calculate_funding_revenue(self, ctx: MetricsContext) -> float:
        """Calculate total funding revenue from funding payments."""
        if not ctx.funding_payments:
            return 0.0
        return -sum(payment.get("funding_payment", 0.0) for payment in ctx.funding_payments)

    # ── Duration metrics ───────────────────────────────────────────────

    def _calculate_avg_trade_duration(self, ctx: MetricsContext) -> float:
        """Calculate average trade duration in hours."""
        if len(ctx.trades) < 2:
            return 0.0
        if ctx.use_all_data:
            return self._calculate_avg_duration_accurate(ctx.trades)
        else:
            return self._calculate_avg_duration_approximate(ctx.trades)

    def _calculate_avg_duration_accurate(self, trades: list[dict]) -> float:
        """Calculate accurate average duration by grouping trades by asset."""
        trade_durations = []
        trades_by_asset: dict[str, list] = {}

        for trade in trades:
            asset = trade.get("asset", "")
            if asset not in trades_by_asset:
                trades_by_asset[asset] = []
            trades_by_asset[asset].append(trade)

        for _asset, asset_trades in trades_by_asset.items():
            sorted_trades = sorted(asset_trades, key=lambda t: t.get("time", datetime.min))
            for i in range(1, len(sorted_trades)):
                prev_time = sorted_trades[i - 1].get("time")
                curr_time = sorted_trades[i].get("time")
                if prev_time and curr_time and isinstance(prev_time, datetime) and isinstance(curr_time, datetime):
                    duration = (curr_time - prev_time).total_seconds() / 3600.0
                    if duration > 0:
                        trade_durations.append(duration)

        return sum(trade_durations) / len(trade_durations) if trade_durations else 0.0

    def _calculate_avg_duration_approximate(self, trades: list[dict]) -> float:
        """Calculate approximate average duration for performance."""
        if len(trades) < 2:
            return 0.0
        first_time = trades[0].get("time")
        last_time = trades[-1].get("time")
        if first_time and last_time and isinstance(first_time, datetime) and isinstance(last_time, datetime):
            total_hours = (last_time - first_time).total_seconds() / 3600.0
            if total_hours > 0:
                return total_hours / len(trades)
        return 0.0

    # ── Helpers ────────────────────────────────────────────────────────

    def _get_periods_per_year(self, interval: str) -> float:
        """Calculate number of periods per year for given interval."""
        if interval.endswith("m"):
            minutes = int(interval[:-1])
            return 365 * 24 * 60 / minutes
        elif interval.endswith("h"):
            hours = int(interval[:-1])
            return 365 * 24 / hours
        elif interval.endswith("d"):
            days = int(interval[:-1])
            return 365 / days
        else:
            return 252.0

    def _get_interval_hours(self, interval: str) -> float:
        """Convert interval string to hours."""
        if interval.endswith("m"):
            return int(interval[:-1]) / 60
        elif interval.endswith("h"):
            return float(int(interval[:-1]))
        elif interval.endswith("d"):
            return int(interval[:-1]) * 24
        elif interval.endswith("w"):
            return int(interval[:-1]) * 168
        else:
            return 1.0
