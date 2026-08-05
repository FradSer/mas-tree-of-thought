"""Blind A/B: Workflow engine vs single call, position-swapped judge.

Both arms use the SAME model. A separate judge model (or same, blind) compares
the two answers twice with positions swapped; disagreement = tie. The scorer
prints one number: net workflow wins (workflow_wins - single_wins) across N
problems. Reproduce the README's quality-ablation methodology on the Workflow
engine specifically.

Run: uv run python -m evals.workflow_ablation [--limit N --json out.json]
"""

import argparse
import asyncio
import json
import random

from dialectica import agent_runtime
from dialectica.agent_factory import create_agent
from dialectica.llm_config import get_model_config
from evals.baseline import BASELINE_INSTRUCTION
from evals.judge import BlindJudge, create_judge_agent
from evals.meta_problems import META_PROBLEMS
from evals.problems import DEFAULT_PROBLEMS
from examples.patterns.reflection_pattern import (
    DEFAULT_ANGLES,
    create_reflection_engine,
)

SINGLE = BASELINE_INSTRUCTION  # prompt-matched single call


async def single_arm(a, problem):
    return (await agent_runtime.run_agent(a, SINGLE.format(problem=problem))).strip()


async def workflow_arm(problem):
    model = get_model_config("GENERATOR")
    engine = create_reflection_engine(
        problem,
        angle_models={a: model for a in DEFAULT_ANGLES},
        frame_model=model,
        critique_model=model,
        synthesize_model=model,
    )
    result = await engine.run()
    return result["final_answer"]


async def run(limit, judge_seed):
    # META_PROBLEMS structurally reward the workflow (multi-stakeholder tension,
    # opposing criteria) — the regime where a single linear pass commits to one
    # side. Toggle with WORKFLOW_PROBLEM_SET=default|meta (default: meta).
    import os

    pool = (
        META_PROBLEMS
        if os.environ.get("WORKFLOW_PROBLEM_SET", "meta") != "default"
        else DEFAULT_PROBLEMS
    )
    problems = pool[:limit] if limit else pool
    solver = create_agent(
        role="Generator", role_name="Solver", model_config=get_model_config("GENERATOR")
    )
    judge = BlindJudge(create_judge_agent())
    random.seed(judge_seed)
    rows = []
    wf = sn = ti = 0
    for p in problems:
        w, s = await asyncio.gather(
            workflow_arm(p.statement), single_arm(solver, p.statement)
        )
        # BlindJudge.compare(problem, engine_answer, baseline_answer) internally
        # judges both orders (engine-first, baseline-first) and names the winner
        # "engine"/"baseline" by ARG POSITION. Pass engine first always; the
        # judge's internal position-swap handles position bias. Do NOT shuffle
        # arg order here — shuffling breaks the arg-position naming.
        res = await judge.compare(p.statement, w, s)
        winner = res.winner
        if winner == "engine":
            wf += 1
        elif winner == "baseline":
            sn += 1
        else:
            ti += 1
        rows.append({"id": p.id, "winner": winner})
        print(f"[{p.id}] winner={winner}", flush=True)
    return {"n": len(problems), "workflow": wf, "single": sn, "tie": ti, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--judge-seed", type=int, default=42)
    ap.add_argument("--json", type=str, default="")
    a = ap.parse_args()
    r = asyncio.run(run(a.limit, a.judge_seed))
    print(
        f"\n# workflow={r['workflow']} single={r['single']} tie={r['tie']} (n={r['n']})"
    )
    print(f"# NET workflow wins = {r['workflow'] - r['single']}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f, indent=2)


if __name__ == "__main__":
    main()
