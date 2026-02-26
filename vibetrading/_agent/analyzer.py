"""
BacktestAnalyzer - LLM-powered analysis of backtest results.

Takes backtest output (metrics, trades, equity curve) and uses an LLM to
produce qualitative assessments: strengths, weaknesses, risk evaluation,
and actionable improvement suggestions.

Usage::

    from vibetrading.strategy import analyze

    results = vibetrading.backtest.run(strategy_code)
    report = analyze(results, strategy_code=strategy_code, model="gpt-4o")

    print(report.summary)
    print(report.suggestions)

    # Feed back into generator for iterative improvement
    feedback = report.format_for_llm()
"""

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ANALYSIS_SYSTEM_PROMPT = """You are an expert quantitative trading analyst reviewing backtest results for a cryptocurrency trading strategy.

Your job is to provide a thorough, honest assessment of the strategy's performance. Be specific, data-driven, and actionable.

## Analysis Framework

Evaluate the strategy across these dimensions:
1. **Return & Risk** — Total return, risk-adjusted return (Sharpe), drawdown severity
2. **Trade Quality** — Win rate, average trade duration, trade frequency
3. **Cost Efficiency** — Transaction fees vs returns, funding costs
4. **Robustness** — Consistency of returns, drawdown recovery, liquidation risk

## Scoring Guidelines
- **9-10**: Exceptional. Sharpe > 2.0, drawdown < 10%, strong win rate
- **7-8**: Good. Positive risk-adjusted returns, manageable drawdowns
- **5-6**: Mediocre. Marginal returns or concerning risk metrics
- **3-4**: Poor. Negative returns or extreme drawdowns
- **1-2**: Failing. Liquidation, massive losses, or non-functional

## Output Format

You MUST respond with valid JSON matching this exact structure:
```json
{
    "score": <integer 1-10>,
    "summary": "<2-3 sentence overall assessment>",
    "strengths": ["<strength 1>", "<strength 2>", ...],
    "weaknesses": ["<weakness 1>", "<weakness 2>", ...],
    "risk_assessment": "<1-2 sentence risk evaluation>",
    "suggestions": ["<actionable suggestion 1>", "<actionable suggestion 2>", ...],
    "detailed_analysis": "<3-5 paragraph deep analysis covering return profile, trade behavior, risk characteristics, and market regime sensitivity>"
}
```

Return ONLY valid JSON. No markdown fences, no text before or after."""


@dataclass
class BacktestAnalysisResult:
    """Structured result from LLM-powered backtest analysis.

    Attributes:
        score: Overall strategy score (1-10).
        summary: Brief overall assessment.
        strengths: List of identified strategy strengths.
        weaknesses: List of identified strategy weaknesses.
        risk_assessment: Evaluation of risk characteristics.
        suggestions: Actionable improvement suggestions.
        detailed_analysis: In-depth multi-paragraph analysis.
        raw_metrics: The metrics dict that was analyzed.
    """
    score: int = 0
    summary: str = ""
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    risk_assessment: str = ""
    suggestions: list[str] = field(default_factory=list)
    detailed_analysis: str = ""
    raw_metrics: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        parts = [f"BacktestAnalysisResult(score={self.score}/10)"]
        parts.append(f"  Summary: {self.summary}")
        if self.strengths:
            parts.append("  Strengths:")
            for s in self.strengths:
                parts.append(f"    + {s}")
        if self.weaknesses:
            parts.append("  Weaknesses:")
            for w in self.weaknesses:
                parts.append(f"    - {w}")
        if self.suggestions:
            parts.append("  Suggestions:")
            for i, s in enumerate(self.suggestions, 1):
                parts.append(f"    {i}. {s}")
        return "\n".join(parts)

    def format_for_llm(self) -> str:
        """Format analysis as feedback for LLM strategy regeneration.

        Returns a prompt-ready string that can be appended to a follow-up
        generation request so the LLM can improve the strategy.
        """
        parts = [
            "## Backtest Analysis Feedback\n",
            f"**Overall Score**: {self.score}/10",
            f"**Summary**: {self.summary}\n",
        ]

        if self.weaknesses:
            parts.append("**Weaknesses to Address** (must fix):")
            for i, w in enumerate(self.weaknesses, 1):
                parts.append(f"  {i}. {w}")

        if self.suggestions:
            parts.append("\n**Specific Improvements** (implement these):")
            for i, s in enumerate(self.suggestions, 1):
                parts.append(f"  {i}. {s}")

        if self.strengths:
            parts.append("\n**Strengths to Preserve** (keep these):")
            for i, s in enumerate(self.strengths, 1):
                parts.append(f"  {i}. {s}")

        parts.append(f"\n**Risk Assessment**: {self.risk_assessment}")

        metrics = self.raw_metrics
        if metrics:
            parts.append("\n**Key Metrics**:")
            parts.append(f"  - Total Return: {metrics.get('total_return', 0):.2%}")
            parts.append(f"  - Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            parts.append(f"  - Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            parts.append(f"  - Win Rate: {metrics.get('win_rate', 0):.2%}")
            parts.append(f"  - Total Trades: {metrics.get('number_of_trades', 0)}")

        parts.append(
            "\nPlease regenerate the strategy code addressing all weaknesses "
            "and implementing the suggested improvements while preserving the strengths."
        )
        return "\n".join(parts)


def _prepare_metrics_summary(backtest_results: Dict[str, Any]) -> str:
    """Condense backtest results into a token-efficient summary for LLM."""
    lines = ["## Backtest Metrics\n"]

    metrics = backtest_results.get("metrics", {})
    if metrics:
        total_return = metrics.get("total_return", 0)
        lines.append(f"- Total Return: {total_return:.2%}")
        lines.append(f"- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
        lines.append(f"- Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        lines.append(f"- Win Rate: {metrics.get('win_rate', 0):.2%}")
        lines.append(f"- Number of Trades: {metrics.get('number_of_trades', 0)}")
        lines.append(f"- Avg Trade Duration: {metrics.get('average_trade_duration_hours', 0):.1f}h")
        lines.append(f"- Total TX Fees: ${metrics.get('total_tx_fees', 0):.2f}")
        lines.append(f"- Funding Revenue: ${metrics.get('funding_revenue', 0):.2f}")
        lines.append(f"- Initial Balance: ${metrics.get('initial_balance', 0):,.2f}")
        lines.append(f"- Final Value: ${metrics.get('total_value', 0):,.2f}")
        returns_std = metrics.get("returns_std", 0)
        if returns_std and not (isinstance(returns_std, float) and math.isnan(returns_std)):
            lines.append(f"- Returns Std Dev: {returns_std:.6f}")

    sim = backtest_results.get("simulation_info", {})
    if sim:
        lines.append(f"\n## Simulation Info\n")
        lines.append(f"- Time Range: {sim.get('time_range', 'N/A')}")
        lines.append(f"- Interval: {sim.get('interval', 'N/A')}")
        lines.append(f"- Steps: {sim.get('steps', 0)}")
        lines.append(f"- Liquidated: {sim.get('liquidated', False)}")
        if sim.get("liquidated"):
            lines.append(f"- Liquidation Time: {sim.get('liquidation_time', 'N/A')}")

    return "\n".join(lines)


def _prepare_trade_summary(backtest_results: Dict[str, Any], max_notable: int = 10) -> str:
    """Summarize trades by asset with notable trades sampled."""
    trades = backtest_results.get("trades", [])
    if not trades:
        return "## Trade Summary\n\nNo trades executed."

    lines = [f"## Trade Summary\n\nTotal trades: {len(trades)}\n"]

    by_asset: Dict[str, list] = {}
    for t in trades:
        asset = t.get("asset", "unknown")
        by_asset.setdefault(asset, []).append(t)

    for asset, asset_trades in by_asset.items():
        pnls = [t.get("realized_pnl") or t.get("pnl") for t in asset_trades if t.get("realized_pnl") or t.get("pnl")]
        winning = sum(1 for p in pnls if p and p > 0)
        losing = sum(1 for p in pnls if p and p < 0)

        actions = {}
        for t in asset_trades:
            a = t.get("action", "unknown")
            actions[a] = actions.get(a, 0) + 1

        lines.append(f"### {asset}")
        lines.append(f"  Trades: {len(asset_trades)}")
        lines.append(f"  Actions: {actions}")
        if pnls:
            valid_pnls = [p for p in pnls if p is not None]
            if valid_pnls:
                lines.append(f"  Winning: {winning}, Losing: {losing}")
                lines.append(f"  Total PnL: ${sum(valid_pnls):,.2f}")
                lines.append(f"  Avg PnL: ${sum(valid_pnls)/len(valid_pnls):,.2f}")
                lines.append(f"  Best: ${max(valid_pnls):,.2f}, Worst: ${min(valid_pnls):,.2f}")
        lines.append("")

    notable = _extract_notable_trades(trades, max_notable)
    if notable:
        lines.append("### Notable Trades\n")
        for t in notable:
            time_str = str(t.get("time", ""))[:19]
            pnl = t.get("realized_pnl") or t.get("pnl", 0)
            pnl_str = f"${pnl:+,.2f}" if pnl else "N/A"
            lines.append(
                f"  [{time_str}] {t.get('action', '?')} {t.get('asset', '?')} "
                f"qty={t.get('quantity', 0):.4f} @ ${t.get('price', 0):,.2f} "
                f"PnL={pnl_str}"
            )

    return "\n".join(lines)


def _extract_notable_trades(trades: List[Dict], max_count: int) -> List[Dict]:
    """Pick the most informative trades: largest wins, largest losses, first, last."""
    if not trades:
        return []

    with_pnl = [t for t in trades if t.get("realized_pnl") or t.get("pnl")]
    if not with_pnl:
        return trades[:max_count]

    def get_pnl(t):
        return t.get("realized_pnl") or t.get("pnl") or 0

    sorted_by_pnl = sorted(with_pnl, key=get_pnl)

    notable = set()
    result = []

    if trades:
        for t in [trades[0], trades[-1]]:
            tid = id(t)
            if tid not in notable:
                notable.add(tid)
                result.append(t)

    half = max(1, (max_count - len(result)) // 2)
    for t in sorted_by_pnl[:half]:
        tid = id(t)
        if tid not in notable:
            notable.add(tid)
            result.append(t)
    for t in sorted_by_pnl[-half:]:
        tid = id(t)
        if tid not in notable:
            notable.add(tid)
            result.append(t)

    return result[:max_count]


def _prepare_equity_curve_summary(backtest_results: Dict[str, Any], max_points: int = 30) -> str:
    """Downsample equity curve to key points for LLM context."""
    import pandas as pd

    results_data = backtest_results.get("results")
    if results_data is None or (isinstance(results_data, pd.DataFrame) and results_data.empty):
        return ""

    if isinstance(results_data, pd.DataFrame):
        df = results_data
    elif isinstance(results_data, list) and results_data:
        df = pd.DataFrame(results_data)
    else:
        return ""

    if "total_value" not in df.columns:
        return ""

    lines = ["## Equity Curve (sampled)\n"]

    n = len(df)
    if n <= max_points:
        sampled = df
    else:
        step = max(1, n // max_points)
        indices = list(range(0, n, step))
        if (n - 1) not in indices:
            indices.append(n - 1)
        sampled = df.iloc[indices]

    for _, row in sampled.iterrows():
        ts = str(row.get("timestamp", ""))[:10]
        tv = row.get("total_value", 0)
        lines.append(f"  {ts}: ${tv:,.2f}")

    return "\n".join(lines)


def _build_analysis_prompt(
    backtest_results: Dict[str, Any],
    strategy_code: Optional[str] = None,
    detail_level: str = "standard",
) -> list[dict[str, str]]:
    """Build the message list for LLM analysis."""
    parts = [_prepare_metrics_summary(backtest_results)]
    parts.append(_prepare_trade_summary(
        backtest_results,
        max_notable=20 if detail_level == "detailed" else 10,
    ))

    if detail_level == "detailed":
        curve = _prepare_equity_curve_summary(backtest_results, max_points=50)
        if curve:
            parts.append(curve)

    if strategy_code:
        parts.append(f"## Strategy Code\n\n```python\n{strategy_code}\n```")

    user_content = "\n\n".join(parts)
    user_content += "\n\nAnalyze this backtest and return your assessment as JSON."

    return [
        {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _parse_analysis_response(raw: str) -> dict:
    """Parse LLM JSON response with fallback handling."""
    raw = raw.strip()

    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    import re
    match = re.search(r'\{[\s\S]*\}', raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse LLM analysis response as JSON, using fallback")
    return {
        "score": 5,
        "summary": raw[:500] if raw else "Analysis could not be parsed.",
        "strengths": [],
        "weaknesses": [],
        "risk_assessment": "",
        "suggestions": [],
        "detailed_analysis": raw,
    }


class BacktestAnalyzer:
    """Analyze backtest results using an LLM.

    Wraps any OpenAI-compatible LLM API via ``litellm`` to produce
    structured performance analysis from backtest output.

    Args:
        model: LLM model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514").
        api_key: API key for the model provider.
        temperature: Sampling temperature (default 0.3 for balanced output).
        **kwargs: Additional keyword arguments passed to litellm.completion().

    Example::

        analyzer = BacktestAnalyzer(model="gpt-4o")
        report = analyzer.analyze(backtest_results, strategy_code=code)
        print(report.summary)
        print(report.suggestions)
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        temperature: float = 0.3,
        **kwargs,
    ):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.extra_kwargs = kwargs

    def analyze(
        self,
        backtest_results: Dict[str, Any],
        *,
        strategy_code: Optional[str] = None,
        detail_level: str = "standard",
    ) -> BacktestAnalysisResult:
        """
        Analyze backtest results and return structured assessment.

        Args:
            backtest_results: Dict returned by ``BacktestEngine.run()`` or
                ``vibetrading.backtest.run()``.
            strategy_code: Original strategy source code for joint analysis.
            detail_level: ``"brief"`` (metrics only), ``"standard"`` (metrics +
                trade summary), or ``"detailed"`` (full with equity curve).

        Returns:
            BacktestAnalysisResult with score, strengths, weaknesses, and suggestions.

        Raises:
            ImportError: If litellm is not installed.
            ValueError: If backtest_results is empty or invalid.
        """
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "litellm is required for backtest analysis. "
                "Install it with: pip install 'vibetrading[agent]'"
            )

        if not backtest_results:
            raise ValueError("backtest_results is empty or None.")

        messages = _build_analysis_prompt(
            backtest_results,
            strategy_code=strategy_code,
            detail_level=detail_level,
        )

        completion_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.api_key:
            completion_kwargs["api_key"] = self.api_key
        completion_kwargs.update(self.extra_kwargs)

        response = litellm.completion(**completion_kwargs)
        raw_content = response.choices[0].message.content.strip()

        parsed = _parse_analysis_response(raw_content)

        return BacktestAnalysisResult(
            score=parsed.get("score", 5),
            summary=parsed.get("summary", ""),
            strengths=parsed.get("strengths", []),
            weaknesses=parsed.get("weaknesses", []),
            risk_assessment=parsed.get("risk_assessment", ""),
            suggestions=parsed.get("suggestions", []),
            detailed_analysis=parsed.get("detailed_analysis", ""),
            raw_metrics=backtest_results.get("metrics", {}),
        )


def analyze_backtest(
    backtest_results: Dict[str, Any],
    *,
    strategy_code: Optional[str] = None,
    model: str = "gpt-4o",
    api_key: str | None = None,
    detail_level: str = "standard",
    **kwargs,
) -> BacktestAnalysisResult:
    """
    Convenience function to analyze backtest results in one call.

    Args:
        backtest_results: Dict returned by ``vibetrading.backtest.run()``.
        strategy_code: Original strategy source code for joint analysis.
        model: LLM model identifier (default: "gpt-4o").
        api_key: API key for the model provider.
        detail_level: ``"brief"``, ``"standard"``, or ``"detailed"``.
        **kwargs: Passed to BacktestAnalyzer constructor.

    Returns:
        BacktestAnalysisResult with structured analysis.

    Example::

        import vibetrading.backtest
        from vibetrading.strategy import analyze

        results = vibetrading.backtest.run(code, interval="1h")
        report = analyze(results, strategy_code=code)
        print(report)
    """
    analyzer = BacktestAnalyzer(model=model, api_key=api_key, **kwargs)
    return analyzer.analyze(
        backtest_results,
        strategy_code=strategy_code,
        detail_level=detail_level,
    )
