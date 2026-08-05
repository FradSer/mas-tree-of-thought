"""Reproducible cost-at-fixed-reliability ablation for the repair engine.

This is the project's hard-won metric made permanent. Controlled evals showed a
reasoning scaffold must be judged against MATCHED-COST resampling, not a single
call — otherwise "looks better" fools you. So this runs three arms on a
verifiable problem set and reports BOTH pass-rate AND cost (LLM calls):

  pass@1     : one sample (the single call)
  best-of-K  : K independent samples, pass if any (the resampling control)
  repair@K   : generate -> verify -> repair-against-the-failure (the engine)

The honest questions it answers: does repair beat a single call? does it beat
matched-cost best-of-K on pass-rate? and at equal pass-rate, is it cheaper
(repair short-circuits on success)? It also counts feedback-only wins — repair
fixing via the loop what blind resampling could not.

Live tool: needs a model (OPENAI_API_BASE/_KEY + DEFAULT_/GENERATOR_MODEL_CONFIG).
Not collected by pytest. Run:
  uv run python -m evals.repair_ablation [--k 3] [--limit N] [--json out.json]
"""

import argparse
import asyncio
import json
from collections.abc import Callable

from pydantic import BaseModel, Field

from dialectica import agent_runtime, create_repair_engine
from dialectica.agent_factory import create_agent
from dialectica.llm_config import get_model_config
from dialectica.repair import SOLVE_PROMPT
from evals.code_eval import build_statement, extract_python_code, verify_solution
from evals.code_problems import CodeProblem
from evals.harness import count_agent_calls

CODE_FORMAT = "Return the full implementation in a single ```python code block."


class ProblemAblation(BaseModel):
    """One problem's three-arm outcome with per-arm cost."""

    problem_id: str
    pass_at_1: bool
    best_of_k: bool
    repair: bool
    repair_attempts: int = Field(..., ge=1)
    best_of_k_calls: int = Field(..., ge=0)
    repair_calls: int = Field(..., ge=0)
    feedback_only_win: bool = Field(
        ...,
        description="repair fixed via the loop (attempt>=2) what best-of-K did not.",
    )


class RepairAblationReport(BaseModel):
    """Aggregate pass-rates and costs across all problems."""

    k: int
    results: list[ProblemAblation]

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def pass_at_1(self) -> int:
        return sum(r.pass_at_1 for r in self.results)

    @property
    def best_of_k(self) -> int:
        return sum(r.best_of_k for r in self.results)

    @property
    def repair(self) -> int:
        return sum(r.repair for r in self.results)

    @property
    def feedback_only_wins(self) -> int:
        return sum(r.feedback_only_win for r in self.results)

    @property
    def best_of_k_calls(self) -> int:
        return sum(r.best_of_k_calls for r in self.results)

    @property
    def repair_calls(self) -> int:
        return sum(r.repair_calls for r in self.results)


def code_verifier(problem: CodeProblem) -> Callable[[str], tuple[bool, str]]:
    """A verifier that extracts code from an answer and runs the problem's tests."""

    def verify(answer: str) -> tuple[bool, str]:
        result = verify_solution(problem, extract_python_code(answer))
        return result.passed, result.output

    return verify


async def _ablate_one(problem: CodeProblem, k: int) -> ProblemAblation:
    statement = build_statement(problem)
    verify = code_verifier(problem)
    generator = create_agent(
        role="Generator", role_name="Solver", model_config=get_model_config("GENERATOR")
    )
    prompt = SOLVE_PROMPT.format(problem=statement, format_hint=f" {CODE_FORMAT}")

    # pass@1 and best-of-K share K independent samples; pass@1 is the first.
    with count_agent_calls() as sample_counter:
        samples = await asyncio.gather(
            *[agent_runtime.run_agent(generator, prompt) for _ in range(k)]
        )
    verdicts = [verify(s)[0] for s in samples]

    with count_agent_calls() as repair_counter:
        rr = await create_repair_engine(
            statement, verifier=verify, max_attempts=k, solution_format=CODE_FORMAT
        ).run()

    best = any(verdicts)
    return ProblemAblation(
        problem_id=problem.id,
        pass_at_1=verdicts[0],
        best_of_k=best,
        repair=rr["passed"],
        repair_attempts=rr["attempts"],
        best_of_k_calls=sample_counter.count,
        repair_calls=repair_counter.count,
        feedback_only_win=rr["passed"] and rr["attempts"] >= 2 and not best,
    )


async def run_repair_ablation(
    problems: list[CodeProblem], k: int = 3
) -> RepairAblationReport:
    """Run the three-arm ablation over ``problems`` (sequential, clean cost counts)."""
    results = [await _ablate_one(p, k) for p in problems]
    return RepairAblationReport(k=k, results=results)


def render_markdown(report: RepairAblationReport) -> str:
    k, n = report.k, report.n
    lines = [
        f"# Repair ablation — {n} problems, K={k}",
        "",
        "| arm | pass | LLM calls |",
        "|---|---|---|",
        f"| pass@1 (single call) | {report.pass_at_1}/{n} | {n} |",
        f"| best-of-{k} (resampling) | {report.best_of_k}/{n} | {report.best_of_k_calls} |",
        f"| repair@{k} (engine) | {report.repair}/{n} | {report.repair_calls} |",
        "",
        (
            f"- feedback-only wins (repair fixed via the loop what resampling did not): "
            f"**{report.feedback_only_wins}**"
        ),
        (
            f"- repair vs best-of-{k} at matched cost: "
            f"**{report.repair}/{n} vs {report.best_of_k}/{n}** pass, "
            f"**{report.repair_calls} vs {report.best_of_k_calls}** calls"
        ),
        "",
        "| problem | pass@1 | best-of-K | repair@K | repair attempts |",
        "|---|---|---|---|---|",
    ]
    for r in report.results:
        flag = "  **<- feedback-only**" if r.feedback_only_win else ""
        lines.append(
            f"| {r.problem_id} | {int(r.pass_at_1)} | {int(r.best_of_k)} | "
            f"{int(r.repair)} | {r.repair_attempts}{flag} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair cost/reliability ablation.")
    parser.add_argument("--k", type=int, default=3, help="attempts / samples per arm")
    parser.add_argument("--limit", type=int, default=None, help="first N problems only")
    parser.add_argument("--json", type=str, default=None, help="write JSON report here")
    parser.add_argument(
        "--problems",
        choices=["novel", "hard"],
        default="novel",
        help="benchmark set: novel (medium) or hard",
    )
    args = parser.parse_args()

    if args.problems == "hard":
        from evals.hard_problems import HARD_PROBLEMS as all_problems
    else:
        from evals.novel_problems import NOVEL_PROBLEMS as all_problems

    problems = all_problems[: args.limit] if args.limit else all_problems
    report = asyncio.run(run_repair_ablation(problems, k=args.k))
    print(render_markdown(report))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report.model_dump(), f, indent=2)


if __name__ == "__main__":
    main()
