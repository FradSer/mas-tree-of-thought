"""Honesty-gate ablation for the ensemble search engine — three arms at matched cost.

**Hypothesis H1**: At matched total LLM-call cost, a heterogeneous N-model ensemble
ranked by a ground-truth-grade injected scorer achieves a strictly higher pass-rate
than best-of-N on the single strongest roster member AND the advantage collapses when
the signal is replaced by a constant (no-information) scorer.

Three arms, all call-counted through the single seam (evals.harness.count_agent_calls),
matching the methodology in repair_ablation.py:

  (a) ensemble + signal   — create_ensemble_engine with a real ground-truth scorer
                            (pass/fail verifier wrapped as float: 1.0 or 0.0).
  (b) best-single best-of-N — N independent samples from the strongest single roster
                              member, scored by the same ground-truth scorer; same
                              total call budget N. Implemented as N concurrent
                              run_agent calls (pure resampling, no refinement) so
                              the comparison is the cleanest matched-cost control.
  (c) ensemble blind-pick — same heterogeneous roster, scorer replaced by the
                            constant 0.5. All candidates look equal to the engine,
                            so the first generated candidate is always selected as
                            "best" (no subsequent score > 0.5 = first score). The
                            budget is fully consumed (0.5 < 1.0 = solved_score,
                            no early stop). Isolates whether signal or mere
                            heterogeneity drives (a)'s wins.

KEEP only if (a) > (b) AND (a) > (c). A cheaper (a)≈(b) tie is a secondary note
only (not a substitute for the primary criterion). Otherwise print CUT — consistent
with the dialectic/ToT negative findings documented in the README.

Repair sub-criterion (FR10): also compares multi-model-repair@K vs
single-model-repair@K on the same verifiable set, using history["model"] attribution
to confirm whether wins came from model-switching or same-model resampling.

Matched-cost caveat: "matched" = matched LLM-call COUNT. A flash call is not
dollar-equivalent to a 30B-parameter call; call-count parity is the harness
discipline, identical to repair_ablation.py. The operator adjusts --budget to set N.

Default roster: ["google:gemini-3.5-flash", "openrouter:qwen3.6-32b"]. Two distinct
providers are required for heterogeneity to be real; the ensemble engine warns at
construction if both arms collapse to the same effective model.

Not run in CI. Real run needs GOOGLE_API_KEY + a live multi-provider roster.
Run: uv run python -m evals.ensemble_ablation [--budget N] [--limit P] [--json out.json]
"""

import argparse
import asyncio
import json
from collections.abc import Callable

from pydantic import BaseModel, Field

from dialectica import agent_runtime, create_repair_engine
from dialectica.agent_factory import create_agent
from evals.code_eval import build_statement, extract_python_code, verify_solution
from evals.code_problems import CodeProblem
from evals.harness import count_agent_calls
from examples.patterns.ensemble_pattern import SOLVE_PROMPT, create_ensemble_engine

# Default roster — two distinct models from different providers.
# Matched-cost note: call-count budget is matched across all arms, but a flash
# call is not dollar-equivalent to a 30B-parameter call.
DEFAULT_ROSTER: list[str] = [
    "google:gemini-3.5-flash",
    "openrouter:qwen3.6-32b",
]

# Arm (b) uses the "strongest" single roster member. Defaulting to roster[0].
# For a definitive run, place the model expected to perform best at index 0.
BEST_SINGLE: str = DEFAULT_ROSTER[0]

CODE_FORMAT = "Return the full implementation in a single ```python code block."


class ProblemEnsembleAblation(BaseModel):
    """Per-problem results for all three arms plus the repair sub-criterion."""

    problem_id: str

    # Arm (a): ensemble + ground-truth signal
    ensemble_passed: bool
    ensemble_score: float = Field(..., ge=0.0, le=1.0)
    ensemble_calls: int = Field(..., ge=0)
    ensemble_attempts: int = Field(..., ge=1)

    # Arm (b): N independent samples from the single strongest arm, pick by scorer
    bon_passed: bool
    bon_score: float = Field(..., ge=0.0, le=1.0)
    bon_calls: int = Field(..., ge=0)

    # Arm (c): ensemble + blind scorer (constant 0.5); answer verified separately
    # since the blind scorer never reflects actual quality.
    blind_passed: bool
    blind_calls: int = Field(..., ge=0)
    blind_attempts: int = Field(..., ge=1)

    # Repair sub-criterion: multi-model rotation
    repair_multi_passed: bool
    repair_multi_calls: int = Field(..., ge=0)
    repair_multi_attempts: int = Field(..., ge=1)
    repair_multi_switched: bool = Field(
        ...,
        description=(
            "True when multi-model repair passed AND the winning attempt used a "
            "non-primary arm (model-switching confirmed by history['model'])."
        ),
    )

    # Repair sub-criterion: single-model baseline
    repair_single_passed: bool
    repair_single_calls: int = Field(..., ge=0)
    repair_single_attempts: int = Field(..., ge=1)


class EnsembleAblationReport(BaseModel):
    """Aggregate of all per-problem results across the three arms."""

    budget: int = Field(..., ge=1, description="Max LLM calls per arm per problem.")
    roster: list[str]
    results: list[ProblemEnsembleAblation]

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def pass_a(self) -> int:
        return sum(r.ensemble_passed for r in self.results)

    @property
    def pass_b(self) -> int:
        return sum(r.bon_passed for r in self.results)

    @property
    def pass_c(self) -> int:
        return sum(r.blind_passed for r in self.results)

    @property
    def calls_a(self) -> int:
        return sum(r.ensemble_calls for r in self.results)

    @property
    def calls_b(self) -> int:
        return sum(r.bon_calls for r in self.results)

    @property
    def calls_c(self) -> int:
        return sum(r.blind_calls for r in self.results)

    @property
    def repair_multi_pass(self) -> int:
        return sum(r.repair_multi_passed for r in self.results)

    @property
    def repair_single_pass(self) -> int:
        return sum(r.repair_single_passed for r in self.results)

    @property
    def repair_multi_calls_total(self) -> int:
        return sum(r.repair_multi_calls for r in self.results)

    @property
    def repair_single_calls_total(self) -> int:
        return sum(r.repair_single_calls for r in self.results)

    @property
    def model_switched_wins(self) -> int:
        return sum(r.repair_multi_switched for r in self.results)


def _code_verifier(problem: CodeProblem) -> Callable[[str], tuple[bool, str]]:
    """Repair-engine verifier: extract code from the answer and run the tests."""

    def verify(answer: str) -> tuple[bool, str]:
        result = verify_solution(problem, extract_python_code(answer))
        return result.passed, result.output

    return verify


def _code_scorer(problem: CodeProblem) -> Callable[[str], float]:
    """Float scorer wrapping the ground-truth verifier: 1.0 = pass, 0.0 = fail."""
    verify = _code_verifier(problem)

    def score(answer: str) -> float:
        passed, _ = verify(answer)
        return 1.0 if passed else 0.0

    return score


async def _ablate_one(
    problem: CodeProblem,
    budget: int,
    roster: list[str],
    best_single: str,
) -> ProblemEnsembleAblation:
    statement = build_statement(problem)
    verifier = _code_verifier(problem)
    scorer = _code_scorer(problem)

    # Arm (a): ensemble + ground-truth signal.
    with count_agent_calls() as cnt_a:
        result_a = await create_ensemble_engine(
            statement,
            scorer,
            models=roster,
            max_calls=budget,
            solution_format=CODE_FORMAT,
        ).run()

    # Arm (b): N independent samples from the single strongest arm.
    # Pure best-of-N (no refinement between samples) is the cleanest matched-cost
    # resampling control; using the same prompt the ensemble uses internally.
    # Parse the config the same way create_ensemble_engine does — create_agent does
    # not parse, so a 'provider:model' string must be resolved (else non-google
    # arms never wrap in LiteLlm and fail to connect).
    from dialectica.llm_config import _parse_model_config

    single_agent = create_agent(
        role="Generator",
        role_name="BestSingle",
        model_config=_parse_model_config(best_single),
    )
    solve_prompt = SOLVE_PROMPT.format(problem=statement, format_hint=f" {CODE_FORMAT}")
    with count_agent_calls() as cnt_b:
        samples_b = await asyncio.gather(
            *[
                agent_runtime.run_agent(single_agent, solve_prompt)
                for _ in range(budget)
            ]
        )
    scores_b = [scorer(s) for s in samples_b]
    best_b_score = max(scores_b)

    # Arm (c): ensemble + blind scorer (constant 0.5).
    # All candidates look equal, so the first generated candidate is always
    # selected (no subsequent score > 0.5 = first score, so best never updates).
    # The full budget is consumed (0.5 < 1.0 = solved_score, no early stop).
    # The answer quality is verified separately since the blind scorer is never
    # an accurate measure of correctness.
    with count_agent_calls() as cnt_c:
        result_c = await create_ensemble_engine(
            statement,
            lambda _: 0.5,
            models=roster,
            max_calls=budget,
            solution_format=CODE_FORMAT,
        ).run()
    blind_passed, _ = verifier(result_c["final_answer"])

    # Repair sub-criterion: multi-model rotation vs single-model.
    with count_agent_calls() as cnt_rm:
        result_rm = await create_repair_engine(
            statement,
            verifier=verifier,
            max_attempts=budget,
            models=roster,
            solution_format=CODE_FORMAT,
        ).run()

    # Attribute the winning attempt to the model that produced it.
    winning_model: str | None = None
    for entry in result_rm["history"]:
        if entry["passed"]:
            winning_model = entry["model"]
            break
    model_switched_win = result_rm["passed"] and winning_model != roster[0]

    with count_agent_calls() as cnt_rs:
        result_rs = await create_repair_engine(
            statement,
            verifier=verifier,
            max_attempts=budget,
            models=[best_single],
            solution_format=CODE_FORMAT,
        ).run()

    return ProblemEnsembleAblation(
        problem_id=problem.id,
        ensemble_passed=result_a["passed"],
        ensemble_score=result_a["best_score"],
        ensemble_calls=cnt_a.count,
        ensemble_attempts=result_a["attempts"],
        bon_passed=best_b_score >= 1.0,
        bon_score=best_b_score,
        bon_calls=cnt_b.count,
        blind_passed=blind_passed,
        blind_calls=cnt_c.count,
        blind_attempts=result_c["attempts"],
        repair_multi_passed=result_rm["passed"],
        repair_multi_calls=cnt_rm.count,
        repair_multi_attempts=result_rm["attempts"],
        repair_multi_switched=model_switched_win,
        repair_single_passed=result_rs["passed"],
        repair_single_calls=cnt_rs.count,
        repair_single_attempts=result_rs["attempts"],
    )


async def run_ensemble_ablation(
    problems: list[CodeProblem],
    budget: int = 6,
    roster: list[str] | None = None,
    best_single: str | None = None,
) -> EnsembleAblationReport:
    """Run all three arms + repair sub-criterion over ``problems``.

    Sequential (clean per-problem cost counts; no cross-problem seam contamination).
    Prints one line per problem as it completes (matches
    ``ensemble_meta_ablation.py``'s progress style) — at higher budgets a
    single problem can take a long time, and without this a run is a black
    box: there is no way to tell "still grinding on a real multi-attempt
    case" apart from "hung" until the whole report prints at the end.
    """
    _roster = roster if roster is not None else DEFAULT_ROSTER
    _best = best_single if best_single is not None else _roster[0]
    results = []
    for p in problems:
        r = await _ablate_one(p, budget, _roster, _best)
        print(
            f"[{p.id}] signal={int(r.ensemble_passed)}({r.ensemble_calls}c) "
            f"best-of-N={int(r.bon_passed)}({r.bon_calls}c) "
            f"blind={int(r.blind_passed)}({r.blind_calls}c) "
            f"repair-multi={int(r.repair_multi_passed)}({r.repair_multi_calls}c) "
            f"repair-single={int(r.repair_single_passed)}({r.repair_single_calls}c)",
            flush=True,
        )
        results.append(r)
    return EnsembleAblationReport(budget=budget, roster=_roster, results=results)


def render_markdown(report: EnsembleAblationReport) -> str:
    n, budget = report.n, report.budget
    lines = [
        f"# Ensemble ablation (H1) — {n} problems, budget N={budget}",
        "",
        f"Roster: {report.roster}",
        "",
        "## Primary metric: pass-rate at matched call cost",
        "",
        "| arm | pass | LLM calls | description |",
        "|---|---|---|---|",
        (
            f"| (a) ensemble + signal | {report.pass_a}/{n} | {report.calls_a} "
            "| heterogeneous roster + ground-truth scorer |"
        ),
        (
            f"| (b) best-single best-of-{budget} | {report.pass_b}/{n} | {report.calls_b} "
            f"| {budget} independent samples from {report.roster[0]} |"
        ),
        (
            f"| (c) ensemble blind-pick | {report.pass_c}/{n} | {report.calls_c} "
            "| heterogeneous roster, constant scorer (first-pick) |"
        ),
        "",
    ]

    a_beats_b = report.pass_a > report.pass_b
    a_beats_c = report.pass_a > report.pass_c
    a_ties_b = report.pass_a == report.pass_b

    if a_beats_b and a_beats_c:
        lines.append(
            "**KEEP** — (a) > (b) AND (a) > (c): "
            "signal + heterogeneity beats both controls."
        )
    elif a_ties_b and a_beats_c:
        lines.append(
            "**CUT (marginal)** — (a) ties (b) at matched cost; heterogeneity alone "
            "helps vs (c) but the signal adds no pass-rate advantage over resampling. "
            "A cheaper tie with (b) does not meet the KEEP criterion."
        )
    else:
        lines.append(
            "**CUT** — (a) does not strictly beat both (b) and (c). "
            "The ensemble's advantage (if any) is not attributable to the "
            "signal + heterogeneity pairing."
        )

    if a_ties_b and report.calls_a < report.calls_b:
        lines.append(
            f"  Secondary note: (a) reached the same pass-rate in fewer calls "
            f"({report.calls_a} vs {report.calls_b}) — short-circuit advantage "
            f"(H1-cost corollary), but this is not the primary criterion."
        )

    lines += [
        "",
        "## Repair sub-criterion",
        "",
        "| arm | pass | LLM calls | model-switch wins |",
        "|---|---|---|---|",
        (
            f"| multi-model-repair@{budget} | {report.repair_multi_pass}/{n} "
            f"| {report.repair_multi_calls_total} | {report.model_switched_wins} |"
        ),
        (
            f"| single-model-repair@{budget} | {report.repair_single_pass}/{n} "
            f"| {report.repair_single_calls_total} | — |"
        ),
        "",
    ]

    if report.repair_multi_pass > report.repair_single_pass:
        lines.append(
            "**KEEP (repair sub-criterion)** — "
            "multi-model rotation improves on single-model repair."
        )
    else:
        lines.append(
            "**CUT (repair sub-criterion)** — "
            "multi-model repair does not beat single-model; "
            "ship single-model repair only and drop the roster."
        )

    if report.model_switched_wins > 0:
        lines.append(
            f"- {report.model_switched_wins}/{report.repair_multi_pass} multi-model wins "
            f"came from a non-primary arm (model-switching confirmed by history['model'])."
        )
    else:
        lines.append(
            "- No model-switching wins: all multi-model passes came from the primary arm "
            "(rotation added no unique rescue capability)."
        )

    lines += [
        "",
        "## Per-problem detail",
        "",
        (
            f"| problem | (a) signal | (b) best-of-{budget} | (c) blind "
            "| repair-multi | repair-single | switched? |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for r in report.results:
        lines.append(
            f"| {r.problem_id} | {int(r.ensemble_passed)} | {int(r.bon_passed)} "
            f"| {int(r.blind_passed)} | {int(r.repair_multi_passed)} "
            f"| {int(r.repair_single_passed)} | {int(r.repair_multi_switched)} |"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ensemble honesty-gate ablation (H1): three arms at matched LLM-call cost + "
            "repair sub-criterion. Needs GOOGLE_API_KEY and a live multi-provider roster. "
            "KEEP only if (a) ensemble+signal > (b) best-single best-of-N AND "
            "(a) > (c) ensemble blind-pick."
        )
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=6,
        help="Max LLM calls per arm per problem (N); all arms share this budget.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run on the first N problems only.",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Write JSON report to this path.",
    )
    parser.add_argument(
        "--problems",
        choices=["novel", "hard"],
        default="novel",
        help="Verifiable benchmark set: novel (medium difficulty) or hard.",
    )
    parser.add_argument(
        "--roster",
        type=str,
        default=None,
        help=(
            "Comma-separated roster of provider:model configs overriding DEFAULT_ROSTER "
            "(e.g. 'openai:qwen3.6-flash,openai:glm-5.2')."
        ),
    )
    parser.add_argument(
        "--best-single",
        type=str,
        default=None,
        help="The roster member used as the best-of-N single-model baseline (arm b).",
    )
    args = parser.parse_args()

    if args.problems == "hard":
        from evals.hard_problems import HARD_PROBLEMS as all_problems
    else:
        from evals.novel_problems import NOVEL_PROBLEMS as all_problems

    problems = all_problems[: args.limit] if args.limit else all_problems
    roster = args.roster.split(",") if args.roster else None
    report = asyncio.run(
        run_ensemble_ablation(
            problems,
            budget=args.budget,
            roster=roster,
            best_single=args.best_single,
        )
    )
    print(render_markdown(report))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report.model_dump(), f, indent=2)


if __name__ == "__main__":
    main()
