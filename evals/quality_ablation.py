"""Does the adversarial best-path structure beat flat compute at the boundary?

The repo's recorded finding is that the engine beats a *naive* single call
(4-1-0) but ties a *prompt-matched* one (0-3-2) — and it openly flags the
missing control: best-of-N + judge. This module supplies the honest opponents.

The thesis under test (the user's): a Tree-of-Thoughts / adversarial engine is a
*workflow* that squeezes the model's capability via opposition + best-path
selection, raising answer quality at the boundary of what the model can do. If
that is the tree *structure* doing the work, the engine must beat — at MATCHED
compute, on quality-graded boundary tasks, under a blind position-swapped judge:

  1. prompt-matched single call      (1 call)   — is the gain just a good prompt?
  2. best-of-N + selector            (~C calls) — is it just parallel sampling + pick?
  3. K-round self-refine             (~C calls) — is it just opposition, no tree?

where C = the engine's own call count for that problem. Every method's calls are
counted through the ``run_agent`` seam (``harness.count_agent_calls``), so
"matched compute" is measured, not assumed. The judge runs on a separate model
and its calls are measurement, not counted against any contender — exactly as in
the existing harness.

Reading the result:
  * engine beats BOTH best-of-N and self-refine  -> the structure adds value;
    the thesis holds and a faithful rebuild is justified.
  * engine ties them but all three beat the single call -> the gain is
    compute + opposition, not the tree; ship the simpler best-of-N/self-refine.
  * nothing beats the single call -> the tasks are saturated; need harder ones.

Live (needs a model). Run:
    uv run python -m evals.quality_ablation [--engine tot|dialectic --limit N --json out.json]
"""

import argparse
import asyncio
import json
import re
import time

from dialectica import agent_runtime
from dialectica.agent_factory import create_agent
from dialectica.llm_config import get_model_config
from examples.patterns.dialectic_pattern import create_dialectic_engine
from examples.patterns.tot_gan_pattern import create_engine

from .baseline import SingleCallBaseline, create_baseline_agent
from .harness import count_agent_calls
from .judge import BlindJudge, create_judge_agent
from .problems import DEFAULT_PROBLEMS

# --- Matched-compute controls --------------------------------------------

_SELECTOR_PROMPT = """You are picking the single best solution to a problem.

**Problem:**
{problem}

**Candidate solutions:**
{candidates}

Judge only merit: correctness, completeness, specificity, actionability.
Reply with ONLY the number of the best candidate (e.g. "3"), nothing else."""

_CRITIQUE_PROMPT = """Critique this solution to the problem rigorously.

**Problem:**
{problem}

**Solution:**
{solution}

List its most important concrete weaknesses and exactly what would make it
stronger — specificity, missing steps, wrong trade-offs, infeasibilities.
Be concrete and actionable; do not rewrite the solution."""

_IMPROVE_PROMPT = """Improve the solution using the critique.

**Problem:**
{problem}

**Current solution:**
{solution}

**Critique:**
{critique}

Produce a single, stronger, complete solution that fixes the weaknesses while
keeping what worked. Provide the solution directly, no commentary."""


async def best_of_n(agent, problem: str, n: int) -> str:
    """Sample ``n`` independent answers, then have the model pick the best.

    Flat parallel sampling + selection — the same model, no tree, no spiral.
    Total cost: ``n`` generation calls + 1 selector call.
    """
    from .baseline import BASELINE_INSTRUCTION

    answers = await asyncio.gather(
        *(
            agent_runtime.run_agent(agent, BASELINE_INSTRUCTION.format(problem=problem))
            for _ in range(n)
        )
    )
    answers = [a.strip() for a in answers]
    if len(answers) == 1:
        return answers[0]

    candidates = "\n\n".join(f"[{i + 1}]\n{a}" for i, a in enumerate(answers))
    pick = await agent_runtime.run_agent(
        agent, _SELECTOR_PROMPT.format(problem=problem, candidates=candidates)
    )
    m = re.search(r"\d+", pick)
    idx = (int(m.group()) - 1) if m else 0
    return answers[idx] if 0 <= idx < len(answers) else answers[0]


async def self_refine(agent, problem: str, rounds: int) -> str:
    """One path, ``rounds`` of critique -> improve. Opposition without a tree.

    Total cost: 1 draft + 2 calls per round.
    """
    from .baseline import BASELINE_INSTRUCTION

    solution = (
        await agent_runtime.run_agent(
            agent, BASELINE_INSTRUCTION.format(problem=problem)
        )
    ).strip()
    for _ in range(rounds):
        critique = await agent_runtime.run_agent(
            agent, _CRITIQUE_PROMPT.format(problem=problem, solution=solution)
        )
        solution = (
            await agent_runtime.run_agent(
                agent,
                _IMPROVE_PROMPT.format(
                    problem=problem, solution=solution, critique=critique
                ),
            )
        ).strip()
    return solution


# --- Runner ---------------------------------------------------------------


def _engine_factory(kind: str):
    if kind == "dialectic":
        # Pin max_rounds=3: the 0-3-2 finding this ablation reproduces was
        # measured on the 3-round dialectic. The pattern's default is now 5
        # (the finding-#9 tuning); leaving it unpinned would silently re-measure
        # a different configuration than the one the 0-3-2 citation describes.
        return lambda stmt: create_dialectic_engine(stmt, max_rounds=3)
    return lambda stmt: create_engine(stmt, max_depth=2, beam_width=2, max_gan_rounds=2)


async def run(engine_kind: str, limit: int) -> dict:
    problems = DEFAULT_PROBLEMS[:limit] if limit else DEFAULT_PROBLEMS
    engine_factory = _engine_factory(engine_kind)
    solver = create_agent(
        role="Generator", role_name="Solver", model_config=get_model_config("GENERATOR")
    )
    single = SingleCallBaseline(create_baseline_agent())
    judge = BlindJudge(create_judge_agent())

    rows = []
    for p in problems:
        with count_agent_calls() as ec:
            t0 = time.perf_counter()
            engine_answer = (await engine_factory(p.statement).run())["final_answer"]
            engine_seconds = time.perf_counter() - t0
        c = ec.count
        # Match the engine's call count, but cap it. The engine's calls are
        # mostly SHORT (strategy lists, scores); each best-of-N / self-refine
        # call is a FULL answer. An uncapped match hands the flat baselines
        # several times the engine's *token* budget (and buries free-tier TPM),
        # so capping is both tractable and fairer. Call counts are reported, so
        # the (under-)matching is transparent.
        n = max(2, min(8, c - 1))
        rounds = max(1, min(3, (c - 1) // 2))

        with count_agent_calls() as bc:
            single_answer = await single.answer(p.statement)
        with count_agent_calls() as nc:
            bon_answer = await best_of_n(solver, p.statement, n)
        with count_agent_calls() as rc:
            sr_answer = await self_refine(solver, p.statement, rounds)

        vs_single = await judge.compare(p.statement, engine_answer, single_answer)
        vs_bon = await judge.compare(p.statement, engine_answer, bon_answer)
        vs_sr = await judge.compare(p.statement, engine_answer, sr_answer)

        rows.append(
            {
                "id": p.id,
                "engine_calls": c,
                "single_calls": bc.count,
                "best_of_n_calls": nc.count,
                "self_refine_calls": rc.count,
                "n": n,
                "rounds": rounds,
                "engine_seconds": round(engine_seconds, 1),
                "vs_single": vs_single.winner,
                "vs_best_of_n": vs_bon.winner,
                "vs_self_refine": vs_sr.winner,
            }
        )
        print(
            f"[{p.id}] engine={c}c/{engine_seconds:.0f}s  "
            f"vs_single={vs_single.winner}  vs_bon={vs_bon.winner}  "
            f"vs_sr={vs_sr.winner}",
            flush=True,
        )

    return {"engine": engine_kind, "n_problems": len(problems), "rows": rows}


def _tally(rows, key: str) -> tuple[int, int, int]:
    win = sum(r[key] == "engine" for r in rows)
    loss = sum(r[key] == "baseline" for r in rows)
    tie = sum(r[key] == "tie" for r in rows)
    return win, tie, loss


def render(result: dict) -> str:
    rows = result["rows"]
    lines = [
        f"# Quality ablation | engine={result['engine']} | "
        "blind position-swapped judge, matched compute\n",
        f"{'problem':>16}  {'eng':>4} {'bon':>4} {'sr':>4}  "
        "vs_single  vs_best_of_n  vs_self_refine",
    ]
    for r in rows:
        lines.append(
            f"{r['id']:>16}  {r['engine_calls']:>4} {r['best_of_n_calls']:>4} "
            f"{r['self_refine_calls']:>4}  {r['vs_single']:>9}  "
            f"{r['vs_best_of_n']:>12}  {r['vs_self_refine']:>14}"
        )
    lines.append("\n# engine W-T-L (matched compute):")
    for label, key in [
        ("vs prompt-matched single", "vs_single"),
        ("vs best-of-N + selector ", "vs_best_of_n"),
        ("vs K-round self-refine   ", "vs_self_refine"),
    ]:
        w, t, ls = _tally(rows, key)
        lines.append(f"  {label}: {w}-{t}-{ls}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Adversarial engine vs matched-compute flat baselines."
    )
    p.add_argument("--engine", choices=["tot", "dialectic"], default="tot")
    p.add_argument(
        "--limit", type=int, default=0, help="Only run the first N problems."
    )
    p.add_argument(
        "--json", type=str, default="", help="Write raw results to this path."
    )
    args = p.parse_args()

    result = asyncio.run(run(args.engine, args.limit))
    print(render(result))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
