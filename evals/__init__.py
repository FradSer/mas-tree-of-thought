"""Eval harness for Dialectica — shared primitives for the live ablation scripts.

A development tool, not part of the published package. The current evals are
the ablation scripts documented in the README (``reflection_ablation``,
``workflow_ablation``, ``quality_workflow_ablation``); this package only holds
the shared machinery they build on (judge, baseline, problem sets).

The historical ``python -m evals`` CLI (ToT+GAN engine vs single-call baseline)
was removed; the three ablations above are the current methodology.
"""

from .baseline import BASELINE_INSTRUCTION
from .harness import count_agent_calls
from .judge import BlindJudge, create_judge_agent
from .problems import DEFAULT_PROBLEMS, EvalProblem

__all__ = [
    "BASELINE_INSTRUCTION",
    "DEFAULT_PROBLEMS",
    "BlindJudge",
    "EvalProblem",
    "count_agent_calls",
    "create_judge_agent",
]
