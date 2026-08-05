"""Blind A/B: edited pure-LLM dialectic vs PROMPT-MATCHED strong single call.

SCORED design (2026-08-05): the original blind win/lose/tie binary NET swung
+-4 between runs of the SAME code (0-3-0 => NET -3, then 2-1-0 => NET +1),
because a discrete head-to-head pick over 3 problems is dominated by judge and
answer-generation stochasticity. This version uses a CONTINUOUS score: the judge
grades each answer 0-10 against DEFAULT_CRITERIA, and the score is
mean(dialectic) - mean(matched baseline) across N problems. Same-model, same
judge, blind to which answer came from where.

The LAST stdout line is a single float: NET = mean(dialectic - matched) across N
meta problems. Autoresearch reads that line as the score (direction=max).

Reference arms (also scored, printed for interpretation):
  - dialectic vs NAIVE single call — the known trap (was 4-1-0 on wins).
  - hetero reflection vs MATCHED baseline — the model-independence lever.

Model wiring mirrors ``evals/score_workflow.sh``: reads ``CLIPROXYAPI_*``,
defaults ``DEFAULT_MODEL_CONFIG=openai:qwen3.6-35b-a3b`` (same model for the
scaffold AND both single baselines — the pure-LLM condition) and
``JUDGE_MODEL_CONFIG=openai:gpt-5.5`` (fast enough for the autoresearch loop's
30-min trial budget; a stronger frontier judge (claude-opus-class) was measured
at ~40s/call and blew the budget; a fast weak judge (gpt-5.4-mini) failed to
discriminate. gpt-5.5 gives ~5s/scored-call.)

Run with --limit 3 for the autoresearch loop (3 problems ≈ 15-20 min with gpt-5.5,
under the 30-min trial timeout).

Run: uv run python -m evals.scaffold_boundary [--limit N --json out.json]
"""

import argparse
import asyncio
import json
import os

from google.adk.agents import LlmAgent

from dialectica import agent_runtime
from dialectica.agent_factory import create_agent
from dialectica.json_repair import strip_code_fence
from dialectica.llm_config import get_model_config
from evals.baseline import BASELINE_INSTRUCTION
from evals.harness import count_agent_calls
from evals.meta_problems import META_PROBLEMS
from examples.patterns._scoring import (
    DEFAULT_CRITERIA,
    Verdict,
    build_scoring_prompt,
    clamp_score,
)
from examples.patterns.dialectic_pattern import create_dialectic_engine
from examples.patterns.reflection_pattern import create_reflection_engine

# The PROMPT-MATCHED strong single call: the same quality bar the dialectic's own
# synthesis prompt holds its output to ("DOMINATE what a single expert writes on
# a first pass") plus the same DEFAULT_CRITERIA. Same model as the scaffold.
STRONG_BASELINE_SYSTEM = """You are an expert problem solver.

Give your single best answer to the problem — the complete answer an expert
would deliver: correct, comprehensive, specific, and actionable, structured with
clear sections where it helps.

Your answer must DOMINATE what a single expert writes on a first pass — that is
the bar it is measured against:
- Be at least as complete and concrete as the better individual solution: carry
  forward specific, actionable detail (numbers, steps, sequencing). Do NOT
  abstract the specifics away into generalities.
- Make a clear, decisive recommendation, and state the precise conditions under
  which the opposite choice would win instead.
- Name the failure mode of the naive one-sided answer, and show concretely how
  your answer avoids it.

**What counts as a strong solution (hold yourself to this):**
{criteria}"""

STRONG_BASELINE_INSTRUCTION = """Solve the following problem:

**Problem:**
{problem}

Provide the solution directly."""


def _wire_env() -> None:
    """Default the cliproxy wiring like ``score_workflow.sh`` (never clobber set vars).

    ``CLIPROXYAPI_HOST`` is a bare host (no scheme/port); wrap it the same way
    ``evals/score_workflow.sh`` does. No private-IP fallback: a LAN address must
    not be baked in as a default, since this file lives in a public repo.
    """
    host = os.environ.get("CLIPROXYAPI_HOST")
    if host:
        os.environ.setdefault(
            "OPENAI_API_BASE",
            f"http://{host}:{os.environ.get('CLIPROXYAPI_HOST_PORT', '8317')}/v1",
        )
    os.environ.setdefault("OPENAI_API_KEY", os.environ.get("CLIPROXYAPI_TOKEN", ""))
    os.environ.setdefault("DEFAULT_MODEL_CONFIG", "openai:qwen3.6-35b-a3b")
    os.environ.setdefault("JUDGE_MODEL_CONFIG", "openai:gpt-5.5")
    os.environ.setdefault("DIALECTICA_DISABLE_THINKING", "true")
    os.environ.setdefault("DIALECTICA_WORKFLOW_CONCURRENCY", "2")


async def dialectic_arm(problem: str) -> str:
    """The edited pure-LLM scaffold, same default model for every stage."""
    engine = create_dialectic_engine(problem)  # model_config=None -> GENERATOR model
    return (await engine.run())["final_answer"].strip()


async def reflection_arm(problem: str) -> str:
    """Reference: hetero reflection — the model-independence lever.

    Uses an explicit two-model roster of WORKING cliproxy ids. The pattern's
    ``DEFAULT_ROSTER`` hardcodes ``openai:qwen3.6-flash``/``openai:glm-5.2``,
    which this proxy no longer serves (502 unknown provider) — so pass working
    ids explicitly. Heterogeneity is the point: it adds information a single
    pass lacks, which is the alternative explanation for any boundary crossing.
    """
    engine = create_reflection_engine(
        problem,
        roster=["openai:qwen3.6-35b-a3b", "openai:deepseek-v4-flash"],
    )
    return (await engine.run())["final_answer"].strip()


def create_strong_baseline_agent() -> LlmAgent:
    """Single call given the SAME quality bar as the scaffold's synthesis bar."""
    return LlmAgent(
        name="StrongBaseline",
        instruction=STRONG_BASELINE_SYSTEM.format(criteria=DEFAULT_CRITERIA),
        model=get_model_config("GENERATOR"),
    )


async def matched_arm(problem: str, strong_solver: LlmAgent) -> str:
    return (
        await agent_runtime.run_agent(
            strong_solver, STRONG_BASELINE_INSTRUCTION.format(problem=problem)
        )
    ).strip()


async def naive_arm(problem: str, solver) -> str:
    return (
        await agent_runtime.run_agent(
            solver, BASELINE_INSTRUCTION.format(problem=problem)
        )
    ).strip()


async def score_answer(problem: str, answer: str) -> float:
    """Judge grades the answer 0-10 against DEFAULT_CRITERIA (blind, continuous).

    Uses ``agent_runtime.run_agent`` directly (not ``wf.agent``, which requires a
    Workflow script context) — the same seam the repo's own ``judge.py`` uses.
    The judge's prompt-driven JSON is parsed into a ``Verdict``; unparseable
    output re-asks (up to 3), then defaults to the neutral midpoint 5.0 — never
    0.0, which would credit the opposing arm with a full 10-point win on that
    problem and manufacture a NET swing larger than the signal being measured
    (the repo's ``judge.py`` maps a ``parse_failed`` verdict to a neutral tie,
    not an extreme score).
    """
    instruction = build_scoring_prompt(answer, {"problem": problem}, DEFAULT_CRITERIA)
    verdict: Verdict | None = None
    for _ in range(3):
        raw = (await agent_runtime.run_agent(_judge_agent(), instruction)).strip()
        body = strip_code_fence(raw)
        try:
            data = json.loads(body)
            verdict = Verdict.model_validate(data)
        except Exception:
            continue
        if verdict is not None:
            return clamp_score(verdict)
    return 5.0


_judge_agent_cache: LlmAgent | None = None


def _judge_agent() -> LlmAgent:
    """A plain scoring judge (criteria live in the prompt, not the system prompt)."""
    global _judge_agent_cache
    if _judge_agent_cache is None:
        _judge_agent_cache = LlmAgent(
            name="Judge",
            instruction="You are a rigorous critic. Grade each answer on the "
            "criteria given, returning only a single JSON object.",
            model=get_model_config("Judge"),
        )
    return _judge_agent_cache


async def run(limit: int | None) -> dict:
    _wire_env()
    problems = META_PROBLEMS[:limit] if limit else META_PROBLEMS
    solver = create_agent(
        role="Generator", role_name="Solver", model_config=get_model_config("GENERATOR")
    )
    strong_solver = create_strong_baseline_agent()

    # NET = mean over problems of (dialectic_score - matched_score). Continuous.
    nets = {"dial_matched": [], "dial_naive": [], "refl_matched": []}
    rows: list[dict] = []

    for p in problems:
        with count_agent_calls() as counter:
            dial, refl, matched, naive = await asyncio.gather(
                dialectic_arm(p.statement),
                reflection_arm(p.statement),
                matched_arm(p.statement, strong_solver),
                naive_arm(p.statement, solver),
            )
            # Score all four answers against DEFAULT_CRITERIA (blind, no attribution).
            s_dial, s_refl, s_matched, s_naive = await asyncio.gather(
                score_answer(p.statement, dial),
                score_answer(p.statement, refl),
                score_answer(p.statement, matched),
                score_answer(p.statement, naive),
            )
        row: dict = {
            "id": p.id,
            "calls": counter.count,
            "scores": {
                "dial": round(s_dial, 2),
                "refl": round(s_refl, 2),
                "matched": round(s_matched, 2),
                "naive": round(s_naive, 2),
            },
            "nets": {
                "dial_matched": round(s_dial - s_matched, 2),
                "dial_naive": round(s_dial - s_naive, 2),
                "refl_matched": round(s_refl - s_matched, 2),
            },
        }
        for k, v in nets.items():
            v.append(row["nets"][k])
        rows.append(row)
        print(
            f"[{p.id}] dial={s_dial:.2f} matched={s_matched:.2f} "
            f"dial_matched_net={s_dial - s_matched:+.2f} "
            f"calls={counter.count}",
            flush=True,
        )

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "n": len(problems),
        "roster_dial": "same-model (GENERATOR default)",
        "nets": {k: mean(v) for k, v in nets.items()},
        "rows": rows,
    }


def render(report: dict) -> str:
    n = report["n"]
    lines = [f"# scaffold boundary (scored) — {n} meta problems"]
    for label, title in (
        ("dial_matched", "dialectic vs MATCHED strong single (PRIMARY)"),
        ("dial_naive", "dialectic vs NAIVE single (known trap, reference)"),
        (
            "refl_matched",
            "hetero reflection vs MATCHED single (independence, reference)",
        ),
    ):
        lines.append(
            f"## {title}\nNET (mean score diff) = {report['nets'][label]:+.3f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pure-LLM scaffold vs prompt-matched single call (scored, boundary hunt)."
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()
    report = asyncio.run(run(args.limit or None))
    print(render(report))
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
    # LAST stdout line = the score autoresearch reads (direction=max).
    print(f"{report['nets']['dial_matched']:.3f}")


if __name__ == "__main__":
    main()
