"""High-concurrency scale harness for the access-list reflection recipe.

Fans out N concurrent ``Workflow`` runs of ``create_reflection_engine`` (with
``use_access_lists=True``) over a problem pool against any OpenAI-compatible
endpoint, measuring:

  * total wall-clock and aggregate LLM calls/second (throughput)
  * per-run latency distribution (p50 / p95 / max)
  * success vs failure vs null-result rate
  * total and per-run LLM call counts (cost proxy)

Env-driven (see CLAUDE.md "Gotchas"), so the SAME harness targets an internal
NAS endpoint with no code change:

    export OPENAI_API_BASE=http://10.10.0.195:<port>/v1
    export OPENAI_API_KEY=<token>
    export DEFAULT_MODEL_CONFIG=openai:<model-name-on-NAS>
    export DIALECTICA_DISABLE_THINKING=true   # for Qwen/thinking models
    uv run python -m evals.access_list_scale --problems 50 --concurrency 8

NOT run in CI; NOT shipped in the wheel (``evals/`` is a dev tool). The numbers
this emits are real end-to-end measurements against a live model — it does NOT
fabricate throughput or latency.

Scale knobs:
  --problems N    how many problem-instances to run (the pool is cycled if N >
                  the builtin problem set; total runs = --problems × --repeats)
  --concurrency C max concurrent Workflow runs (the kernel's own per-run
                  concurrency cap is separate and still gates agent() calls)
  --repeats R     repetitions per problem (for variance)
  --access-lists  default True; pass --no-access-lists to compare the inlined
                  (measured) prompt path head-to-head at the same scale.
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from evals.harness import count_agent_calls
from evals.meta_problems import META_PROBLEMS
from evals.problems import DEFAULT_PROBLEMS
from examples.patterns.reflection_pattern import create_reflection_engine

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover — dotenv is a dev dependency in evals

    def load_dotenv(_path):
        return False


def _problem_pool(n: int, use_meta: bool) -> list[str]:
    base = META_PROBLEMS if use_meta else DEFAULT_PROBLEMS
    pool = [p.statement for p in base]
    if n <= len(pool):
        return pool[:n]
    # Cycle the pool to reach the requested instance count.
    return [pool[i % len(pool)] for i in range(n)]


async def _one_run(
    statement: str, roster: list[str], use_access_lists: bool, runtime: list
) -> tuple[str, int, float, str | None]:
    """Run one reflection engine; record (label, calls, latency_seconds, error)."""
    engine = create_reflection_engine(
        statement, roster=roster, use_access_lists=use_access_lists
    )
    start = time.perf_counter()
    try:
        with count_agent_calls() as counter:
            result = await engine.run()
    except Exception as e:  # harness must record, not crash
        latency = time.perf_counter() - start
        runtime.append({"latency": latency, "calls": 0, "ok": False, "err": str(e)})
        return ("error", 0, latency, str(e))
    latency = time.perf_counter() - start
    final = result.get("final_answer", "")
    ok = bool(final and final.strip() and len(final.strip()) > 40)
    runtime.append(
        {"latency": latency, "calls": counter.count, "ok": ok, "len": len(final)}
    )
    return (final.strip() if final else "", counter.count, latency, None)


def _pctile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1, round((pct / 100.0) * (len(sorted_vals) - 1))))
    return sorted_vals[k]


async def run(
    n: int,
    concurrency: int,
    repeats: int,
    use_access_lists: bool,
    use_meta: bool,
    roster: list[str],
) -> dict:
    statements = _problem_pool(n, use_meta)
    # Build the run queue: each problem × repeats.
    queue = []
    for r in range(repeats):
        for i, s in enumerate(statements):
            queue.append((f"{i}-{r}", s))
    total_runs = len(queue)
    sem = asyncio.Semaphore(concurrency)
    runtime: list[dict] = []

    async def gated(label: str, statement: str) -> None:
        async with sem:
            await _one_run(statement, roster, use_access_lists, runtime)

    wall_start = time.perf_counter()
    await asyncio.gather(*[gated(lbl, s) for lbl, s in queue])
    wall = time.perf_counter() - wall_start

    latencies = sorted(r["latency"] for r in runtime)
    call_counts = [r["calls"] for r in runtime]
    oks = sum(1 for r in runtime if r["ok"])
    fails = sum(1 for r in runtime if not r["ok"])
    total_calls = sum(call_counts)
    return {
        "endpoint": os.environ.get("OPENAI_API_BASE", "(unset)"),
        "model": os.environ.get("DEFAULT_MODEL_CONFIG", "(unset)"),
        "mode": "access_lists" if use_access_lists else "inlined",
        "total_runs": total_runs,
        "concurrency": concurrency,
        "wall_clock_seconds": round(wall, 2),
        "total_llm_calls": total_calls,
        "calls_per_second": round(total_calls / wall, 2) if wall else 0.0,
        "runs_per_second": round(total_runs / wall, 4) if wall else 0.0,
        "success_rate": round(oks / total_runs, 4) if total_runs else 0.0,
        "failures": fails,
        "latency_p50_seconds": round(_pctile(latencies, 50), 3),
        "latency_p95_seconds": round(_pctile(latencies, 95), 3),
        "latency_max_seconds": round(max(latencies) if latencies else 0.0, 3),
        "avg_calls_per_run": round(total_calls / total_runs, 2) if total_runs else 0.0,
        "roster": roster,
    }


def render(report: dict) -> str:
    return f"""# Access-list reflection — scale run

Endpoint: {report["endpoint"]}
Model: {report["model"]}
Mode: {report["mode"]}  |  Roster: {report["roster"]}

## Throughput
| metric | value |
|---|---|
| total runs | {report["total_runs"]} |
| concurrency | {report["concurrency"]} |
| wall-clock (s) | {report["wall_clock_seconds"]} |
| total LLM calls | {report["total_llm_calls"]} |
| LLM calls/sec | {report["calls_per_second"]} |
| runs/sec | {report["runs_per_second"]} |

## Reliability & cost
| metric | value |
|---|---|
| success rate | {report["success_rate"]:.1%} |
| failures | {report["failures"]} |
| avg calls/run | {report["avg_calls_per_run"]} |

## Latency per run (seconds)
| p50 | p95 | max |
|---|---|---|
| {report["latency_p50_seconds"]} | {report["latency_p95_seconds"]} | {report["latency_max_seconds"]} |
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="access_list_scale",
        description="High-concurrency scale run of the access-list reflection recipe.",
    )
    p.add_argument(
        "--problems", type=int, default=10, help="distinct problem instances"
    )
    p.add_argument("--concurrency", type=int, default=4, help="max concurrent runs")
    p.add_argument("--repeats", type=int, default=1, help="repetitions per problem")
    p.add_argument(
        "--no-access-lists",
        action="store_true",
        help="use the inlined (measured) prompt path instead of sees= for head-to-head",
    )
    p.add_argument(
        "--default-problems",
        action="store_true",
        help="use the default (non-meta) problem set",
    )
    p.add_argument(
        "--roster",
        type=str,
        default=None,
        help="comma-separated model configs (default: cliproxy glm-5.2/qwen3.6-flash)",
    )
    p.add_argument("--json", type=Path, default=None, help="write the JSON report here")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    load_dotenv(Path(__file__).resolve().parent.parent / "dialectica" / ".env")
    roster = (
        args.roster.split(",")
        if args.roster
        else ["openai:qwen3.6-flash", "openai:glm-5.2"]
    )
    report = await run(
        n=args.problems,
        concurrency=args.concurrency,
        repeats=args.repeats,
        use_access_lists=not args.no_access_lists,
        use_meta=not args.default_problems,
        roster=roster,
    )
    print(render(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"JSON report written to {args.json}")


if __name__ == "__main__":
    asyncio.run(main())
