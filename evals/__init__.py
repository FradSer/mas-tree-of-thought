"""Eval harness for Dialectica — engine vs single-call baseline, judged blind.

A development tool, not part of the published package: it answers "is the
engine worth its cost" with data. Each benchmark problem is solved by the
engine and by a single strong-model call, then a blind LLM judge compares the
two answers (twice, with positions swapped, to neutralize position bias).

Run it with: ``uv run python -m evals``.
"""

from .baseline import SingleCallBaseline, create_baseline_agent
from .harness import (
    EvalReport,
    ProblemResult,
    count_agent_calls,
    render_markdown,
    run_eval,
)
from .judge import BlindJudge, JudgeVerdict, PairwiseResult, create_judge_agent
from .problems import DEFAULT_PROBLEMS, EvalProblem

__all__ = [
    "DEFAULT_PROBLEMS",
    "BlindJudge",
    "EvalProblem",
    "EvalReport",
    "JudgeVerdict",
    "PairwiseResult",
    "ProblemResult",
    "SingleCallBaseline",
    "count_agent_calls",
    "create_baseline_agent",
    "create_judge_agent",
    "render_markdown",
    "run_eval",
]
