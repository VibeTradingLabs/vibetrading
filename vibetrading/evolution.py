"""
Strategy evolution through iterative generate-backtest-analyze loops.

Takes a natural language prompt and iteratively improves the generated strategy
by backtesting, analyzing results with an LLM, and regenerating with feedback.

Usage::

    import vibetrading

    result = vibetrading.evolve(
        "BTC momentum strategy with RSI and ATR stop loss",
        iterations=3,
        model="gpt-4o",
        interval="1h",
    )

    print(result.best_code)
    print(result.best_analysis)
    print(result.history)

    # Or with full control
    from vibetrading.evolution import StrategyEvolver

    evolver = StrategyEvolver(model="gpt-4o")
    result = evolver.evolve("BTC grid strategy", iterations=5)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ._agent.analyzer import (
    BacktestAnalysisResult,
    BacktestAnalyzer,
)
from ._agent.generator import StrategyGenerator
from ._agent.validator import validate_strategy

logger = logging.getLogger(__name__)


@dataclass
class EvolutionStep:
    """Record of a single generate-backtest-analyze iteration.

    Attributes:
        iteration: 0-based iteration index.
        code: Strategy code produced in this iteration.
        validation_passed: Whether static validation passed.
        backtest_results: Raw backtest output dict (None if backtest failed).
        analysis: LLM analysis of backtest results (None if not analyzed).
        error: Error message if this iteration failed.
    """
    iteration: int
    code: str
    validation_passed: bool = False
    backtest_results: Optional[Dict[str, Any]] = None
    analysis: Optional[BacktestAnalysisResult] = None
    error: Optional[str] = None

    @property
    def score(self) -> int:
        if self.analysis:
            return self.analysis.score
        return 0


@dataclass
class EvolutionResult:
    """Result of a multi-iteration strategy evolution.

    Attributes:
        best_code: Strategy code from the highest-scoring iteration.
        best_analysis: Analysis result from the highest-scoring iteration.
        best_backtest: Backtest results from the highest-scoring iteration.
        best_iteration: Index of the best iteration.
        history: Full list of EvolutionStep records.
        prompt: Original natural language prompt.
        total_iterations: Total iterations executed.
    """
    best_code: str = ""
    best_analysis: Optional[BacktestAnalysisResult] = None
    best_backtest: Optional[Dict[str, Any]] = None
    best_iteration: int = -1
    history: List[EvolutionStep] = field(default_factory=list)
    prompt: str = ""
    total_iterations: int = 0

    def __repr__(self) -> str:
        parts = [
            f"EvolutionResult(iterations={self.total_iterations}, "
            f"best_iteration={self.best_iteration})"
        ]
        if self.best_analysis:
            parts.append(f"  Best Score: {self.best_analysis.score}/10")
            parts.append(f"  Summary: {self.best_analysis.summary}")
        parts.append(f"  Scores: {[s.score for s in self.history]}")
        return "\n".join(parts)

    @property
    def best_score(self) -> int:
        if self.best_analysis:
            return self.best_analysis.score
        return 0

    @property
    def best_metrics(self) -> dict:
        """Metrics dict from the best iteration."""
        if self.best_backtest:
            return self.best_backtest.get("metrics", {})
        return {}

    @property
    def improved(self) -> bool:
        """True if later iterations scored higher than the first."""
        scores = [s.score for s in self.history if s.score > 0]
        return len(scores) >= 2 and scores[-1] > scores[0]


class StrategyEvolver:
    """Iteratively improve strategies through generate-backtest-analyze loops.

    Args:
        model: LLM model identifier for both generation and analysis.
        api_key: API key for the LLM provider.
        generator_kwargs: Extra kwargs for StrategyGenerator.
        analyzer_kwargs: Extra kwargs for BacktestAnalyzer.

    Example::

        evolver = StrategyEvolver(model="gpt-4o")
        result = evolver.evolve(
            "BTC momentum with RSI oversold entry",
            iterations=3,
            interval="1h",
        )
        print(result.best_code)
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        generator_kwargs: dict | None = None,
        analyzer_kwargs: dict | None = None,
    ):
        self.model = model
        self.api_key = api_key

        gen_kw = generator_kwargs or {}
        ana_kw = analyzer_kwargs or {}

        self.generator = StrategyGenerator(
            model=model, api_key=api_key, **gen_kw,
        )
        self.analyzer = BacktestAnalyzer(
            model=model, api_key=api_key, **ana_kw,
        )

    def evolve(
        self,
        prompt: str,
        *,
        iterations: int = 3,
        interval: str = "1h",
        initial_balances: Optional[Dict[str, float]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        exchange: str = "binance",
        data: Optional[Dict] = None,
        assets: list[str] | None = None,
        market_type: str | None = None,
        max_leverage: int | None = None,
        detail_level: str = "standard",
        score_threshold: int = 8,
        on_iteration: Any = None,
    ) -> EvolutionResult:
        """
        Run iterative strategy evolution.

        Each iteration: generate/regenerate code -> backtest -> analyze ->
        feed analysis back as context for next generation.

        Args:
            prompt: Natural language strategy description.
            iterations: Maximum number of evolution iterations (default: 3).
            interval: Backtest candle interval (default: "1h").
            initial_balances: Starting balances (default: {"USDC": 10000}).
            start_time: Backtest start time.
            end_time: Backtest end time.
            exchange: Exchange for data lookup (default: "binance").
            data: Pre-loaded data dict.
            assets: Restrict to specific assets.
            market_type: "perp" or "spot".
            max_leverage: Maximum leverage allowed.
            detail_level: Analysis detail ("brief", "standard", "detailed").
            score_threshold: Stop early if score reaches this level (default: 8).
            on_iteration: Optional callback ``fn(step: EvolutionStep)`` called
                after each iteration completes.

        Returns:
            EvolutionResult with best strategy code and full iteration history.
        """
        from .backtest import BacktestEngine

        result = EvolutionResult(prompt=prompt, total_iterations=0)
        feedback_context: str | None = None

        for i in range(iterations):
            logger.info("Evolution iteration %d/%d", i + 1, iterations)
            step = EvolutionStep(iteration=i, code="")

            # --- Generate / Regenerate ---
            try:
                additional_context = feedback_context
                code = self.generator.generate(
                    prompt,
                    assets=assets,
                    market_type=market_type,
                    max_leverage=max_leverage,
                    interval=interval,
                    additional_context=additional_context,
                    validate=True,
                    max_retries=2,
                )
                step.code = code
                step.validation_passed = True
            except ValueError as e:
                step.code = ""
                step.error = f"Generation failed: {e}"
                logger.warning("Iteration %d generation failed: %s", i, e)
                result.history.append(step)
                result.total_iterations = i + 1
                if on_iteration:
                    on_iteration(step)
                continue

            # --- Backtest ---
            try:
                engine = BacktestEngine(
                    interval=interval,
                    initial_balances=initial_balances,
                    start_time=start_time,
                    end_time=end_time,
                    exchange=exchange,
                    data=data,
                    mute_strategy_prints=True,
                )
                bt_results = engine.run(code)
                step.backtest_results = bt_results
            except Exception as e:
                step.error = f"Backtest failed: {e}"
                logger.warning("Iteration %d backtest failed: %s", i, e)
                feedback_context = (
                    f"The previous strategy code caused a backtest error:\n"
                    f"```\n{e}\n```\n"
                    f"Please fix the issue and regenerate."
                )
                result.history.append(step)
                result.total_iterations = i + 1
                if on_iteration:
                    on_iteration(step)
                continue

            # --- Analyze ---
            try:
                analysis = self.analyzer.analyze(
                    bt_results,
                    strategy_code=code,
                    detail_level=detail_level,
                )
                step.analysis = analysis
            except Exception as e:
                step.error = f"Analysis failed: {e}"
                logger.warning("Iteration %d analysis failed: %s", i, e)
                result.history.append(step)
                result.total_iterations = i + 1
                if on_iteration:
                    on_iteration(step)
                continue

            result.history.append(step)
            result.total_iterations = i + 1

            logger.info(
                "Iteration %d: score=%d/10, return=%.2f%%",
                i, analysis.score,
                bt_results.get("metrics", {}).get("total_return", 0) * 100,
            )

            # --- Update best ---
            if result.best_iteration == -1 or analysis.score > result.best_score:
                result.best_code = code
                result.best_analysis = analysis
                result.best_backtest = bt_results
                result.best_iteration = i

            if on_iteration:
                on_iteration(step)

            # --- Early stop ---
            if analysis.score >= score_threshold:
                logger.info(
                    "Score %d reached threshold %d, stopping early",
                    analysis.score, score_threshold,
                )
                break

            # --- Build feedback for next iteration ---
            if i < iterations - 1:
                feedback_context = analysis.format_for_llm()

        return result


def evolve(
    prompt: str,
    *,
    iterations: int = 3,
    model: str = "gpt-4o",
    api_key: str | None = None,
    interval: str = "1h",
    initial_balances: Optional[Dict[str, float]] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    exchange: str = "binance",
    data: Optional[Dict] = None,
    assets: list[str] | None = None,
    market_type: str | None = None,
    max_leverage: int | None = None,
    detail_level: str = "standard",
    score_threshold: int = 8,
    on_iteration: Any = None,
    **kwargs,
) -> EvolutionResult:
    """
    Evolve a trading strategy through iterative LLM-driven improvement.

    This is the main entry point for strategy evolution. It generates a strategy
    from a natural language prompt, backtests it, analyzes the results with an LLM,
    and iteratively improves the code based on the analysis feedback.

    Args:
        prompt: Natural language description of the desired strategy.
        iterations: Maximum evolution iterations (default: 3).
        model: LLM model identifier (default: "gpt-4o").
        api_key: API key for the LLM provider.
        interval: Backtest candle interval (default: "1h").
        initial_balances: Starting balances (default: {"USDC": 10000}).
        start_time: Backtest start time.
        end_time: Backtest end time.
        exchange: Exchange for data lookup (default: "binance").
        data: Pre-loaded data dict mapping "ASSET/interval" to DataFrames.
        assets: Restrict to specific assets (e.g., ["BTC", "ETH"]).
        market_type: "perp" or "spot".
        max_leverage: Maximum leverage allowed.
        detail_level: Analysis detail level ("brief", "standard", "detailed").
        score_threshold: Stop early if score reaches this (default: 8).
        on_iteration: Optional callback ``fn(step: EvolutionStep)`` invoked
            after each iteration.
        **kwargs: Additional kwargs split between generator and analyzer.

    Returns:
        EvolutionResult with best strategy code, analysis, and full history.

    Example::

        import vibetrading

        result = vibetrading.evolve(
            "BTC momentum strategy using RSI and MACD crossover",
            iterations=3,
            model="gpt-4o",
            interval="1h",
            initial_balances={"USDC": 10000},
        )

        print(f"Best score: {result.best_score}/10")
        print(f"Best code:\\n{result.best_code}")
        print(f"Improved: {result.improved}")

        for step in result.history:
            print(f"  Iteration {step.iteration}: score={step.score}")
    """
    evolver = StrategyEvolver(model=model, api_key=api_key)
    return evolver.evolve(
        prompt,
        iterations=iterations,
        interval=interval,
        initial_balances=initial_balances,
        start_time=start_time,
        end_time=end_time,
        exchange=exchange,
        data=data,
        assets=assets,
        market_type=market_type,
        max_leverage=max_leverage,
        detail_level=detail_level,
        score_threshold=score_threshold,
        on_iteration=on_iteration,
    )
