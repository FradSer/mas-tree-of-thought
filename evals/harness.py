"""Eval orchestration: run engine and baseline per problem, judge blind, report.

LLM calls are counted through the single runtime seam
(``dialectica.agent_runtime.run_agent``), so cost is measured the same way the
mocked tests intercept it — judge calls are deliberately not counted against
either contender.
"""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Protocol

from pydantic import BaseModel, Field

from dialectica import agent_runtime

from .baseline import SingleCallBaseline
from .judge import BlindJudge
from .problems import EvalProblem


class EngineLike(Protocol):
    """Structural type for any engine object this harness can run.

    Every engine — shipped (``create_repair_engine``) or a demoted reference
    pattern (``examples/patterns/*.py``) — returns an object with an async
    ``run()``; the harness never needs more than that.
    """

    async def run(self) -> dict[str, Any]: ...


class CallCounter:
    """Counts LLM calls made while a ``count_agent_calls`` block is active."""

    def __init__(self):
        self.count = 0


@contextmanager
def count_agent_calls() -> Iterator[CallCounter]:
    """Count every ``run_agent`` call made inside the block.

    Wraps whatever is currently installed at the seam (the real runner or a
    test fake) and restores it on exit.
    """
    counter = CallCounter()
    original = agent_runtime.run_agent

    async def counting_run_agent(agent, instruction: str) -> str:
        counter.count += 1
        return await original(agent, instruction)

    agent_runtime.run_agent = counting_run_agent
    try:
        yield counter
    finally:
        agent_runtime.run_agent = original


class ProblemResult(BaseModel):
    """Engine-vs-baseline outcome for one benchmark problem."""

    problem_id: str
    problem: str
    engine_answer: str
    baseline_answer: str
    engine_calls: int = Field(..., ge=0)
    baseline_calls: int = Field(..., ge=0)
    engine_seconds: float = Field(..., ge=0)
    baseline_seconds: float = Field(..., ge=0)
    winner: str = Field(..., description='"engine", "baseline", or "tie".')
    judge_reasoning: list[str] = Field(default_factory=list)


class EvalReport(BaseModel):
    """All per-problem results plus the aggregate tally."""

    results: list[ProblemResult]
    engine_wins: int
    baseline_wins: int
    ties: int

    @classmethod
    def from_results(cls, results: list[ProblemResult]) -> "EvalReport":
        tally = {"engine": 0, "baseline": 0, "tie": 0}
        for result in results:
            tally[result.winner] += 1
        return cls(
            results=results,
            engine_wins=tally["engine"],
            baseline_wins=tally["baseline"],
            ties=tally["tie"],
        )


async def run_eval(
    problems: list[EvalProblem],
    *,
    engine_factory: Callable[[str], EngineLike],
    baseline: SingleCallBaseline,
    judge: BlindJudge,
) -> EvalReport:
    """Run every problem through the engine and the baseline, then judge blind.

    Transient-failure retry lives in ``agent_runtime.run_agent`` itself, so
    the harness measures exactly what the library does in production.
    """
    results: list[ProblemResult] = []
    for problem in problems:
        with count_agent_calls() as engine_counter:
            engine_start = time.perf_counter()
            engine_run = await engine_factory(problem.statement).run()
            engine_seconds = time.perf_counter() - engine_start
        engine_answer = engine_run["final_answer"]

        with count_agent_calls() as baseline_counter:
            baseline_start = time.perf_counter()
            baseline_answer = await baseline.answer(problem.statement)
            baseline_seconds = time.perf_counter() - baseline_start

        comparison = await judge.compare(
            problem.statement, engine_answer, baseline_answer
        )

        results.append(
            ProblemResult(
                problem_id=problem.id,
                problem=problem.statement,
                engine_answer=engine_answer,
                baseline_answer=baseline_answer,
                engine_calls=engine_counter.count,
                baseline_calls=baseline_counter.count,
                engine_seconds=engine_seconds,
                baseline_seconds=baseline_seconds,
                winner=comparison.winner,
                judge_reasoning=[v.reasoning for v in comparison.verdicts],
            )
        )

    return EvalReport.from_results(results)


def render_markdown(report: EvalReport) -> str:
    """Render the report as a human-readable Markdown summary."""
    lines = [
        "# Dialectica eval: engine vs single-call baseline",
        "",
        (
            f"**Engine wins: {report.engine_wins} · "
            f"Baseline wins: {report.baseline_wins} · Ties: {report.ties}**"
        ),
        "",
        "| Problem | Winner | Engine calls | Baseline calls | Engine s | Baseline s |",
        "|---------|--------|--------------|----------------|----------|------------|",
    ]
    for r in report.results:
        lines.append(
            f"| {r.problem_id} | {r.winner} | {r.engine_calls} "
            f"| {r.baseline_calls} | {r.engine_seconds:.1f} | {r.baseline_seconds:.1f} |"
        )
    lines.append("")
    for r in report.results:
        lines.append(f"## {r.problem_id} — winner: {r.winner}")
        for i, reasoning in enumerate(r.judge_reasoning, 1):
            if reasoning:
                lines.append(f"- Judge pass {i}: {reasoning}")
        lines.append("")
    return "\n".join(lines)
