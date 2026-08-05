"""Faithful Princeton-recipe Tree-of-Thoughts on Game-of-24 — saturation check.

This is a STRICT reimplementation of the original ToT recipe
(princeton-nlp/tree-of-thought-llm, Yao et al. 2023) for the Game-of-24 task,
to confirm the saturation finding with the *original* prompts and hyperparameters
rather than my own (`evals/game24.py`). Key faithfulness points vs the original:

  * The LM does its OWN arithmetic and authors its own ``left:`` remaining-set
    — there is NO Python ``apply()``. A wrong ``2 + 8 = 11 (left: 11 8 14)``
    propagates as the next state (original has no mid-search sanity check).
  * propose prompt = the original 1-shot example, verbatim.
  * value prompt = the original 7-shot sure/likely/impossible, verbatim, sent
    N=3 times per state and SUMMED with the original ad-hoc weights
    ``{impossible:0.001, likely:1, sure:20}``.
  * BFS beam b=5, greedy top-b selection, depth 3 intermediate steps.
  * Final answer = the LM's ``Answer: <expr> = 24`` line; correctness judged by
    the shared ``check_24`` sympy-style verifier (same oracle as ``game24.py``).

Saturation hypothesis (2026-06-22): on gemini-3.5-flash the single CoT call
solves ~all puzzles, so faithful-ToT has no gap to recover — confirming that
Game-of-24 is no longer a search-requiring benchmark for modern models.

Baselines at matched cost: a single CoT call (original cot prompt, 5-shot), and
CoT-SC@k (pass if ANY correct). Run:
    GOOGLE_API_KEY=… uv run python -m evals.game24_princeton [--limit N --beam 5 --cot-sc-k K]
"""

import argparse
import asyncio
import re

from dialectica import agent_runtime
from dialectica.agent_factory import create_agent
from dialectica.llm_config import get_model_config

from .game24 import PUZZLES, check_24, extract_expression

# --- Original Princeton prompts (verbatim) -------------------------------
# src/tot/prompts/game24.py — propose (1-shot) and value (7-shot).

PROPOSE_PROMPT = """Input: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 /  2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)
Input: {input}
Possible next steps:
"""

VALUE_PROMPT = """Evaluate if given numbers can reach 24 (sure/likely/impossible)
10 14
10 + 14 = 24
sure
11 12
11 + 12 = 23
12 - 11 = 1
11 * 12 = 132
11 / 12 = 0.91
impossible
4 4 10
4 + 4 + 10 = 8 + 10 = 18
4 * 10 - 4 = 40 - 4 = 36
(10 - 4) * 4 = 6 * 4 = 24
sure
4 9 11
9 + 11 + 4 = 20 + 4 = 24
sure
5 7 8
5 + 7 + 8 = 12 + 8 = 20
(8 - 5) * 7 = 3 * 7 = 21
I cannot obtain 24 now, but numbers are within a reasonable range
likely
5 6 6
5 + 6 + 6 = 17
(6 - 5) * 6 = 1 * 6 = 6
I cannot obtain 24 now, but numbers are within a reasonable range
likely
10 10 11
10 + 10 + 11 = 31
(11 - 10) * 10 = 10
10 10 10 are all too big
impossible
1 3 3
1 * 3 * 3 = 9
(1 + 3) * 3 = 12
1 3 3 are all too small
impossible
{input}
"""

# The original CoT prompt (5-shot) — same as the Princeton cot_prompt.
COT_PROMPT = """Use numbers and basic arithmetic operations (+ - * /) to obtain 24.
Input: 4 4 6 8
4 + 8 = 12 (left: 4 6)
6 * 4 = 24 (left: 12 24)
12 * 24 = 24
Input: 2 9 10 12
2 * 12 = 24 (left: 9 10 24)
24 - 10 = 14 (left: 9 14)
9 + 14 = 23
Input: 4 4 7 8
8 / 4 = 2 (left: 4 7 2)
4 * 7 = 28 (left: 2 28)
28 - 2 = 26
Input: 4 9 10 13
13 - 10 = 3 (left: 4 9 3)
9 * 3 = 27 (left: 4 27)
27 - 4 = 23
Input: 1 3 4 6
6 * 4 = 24 (left: 1 3 24)
3 / 1 = 3 (left: 3 24)
24 * 3 = 72
Input: {input}
"""

# Original ad-hoc value weights (TODO: ad hoc in the source). N value samples
# are SUMMED, not averaged.
_VALUE_MAP = {"impossible": 0.001, "likely": 1.0, "sure": 20.0}


def _current_numbers(trajectory: str) -> str:
    """The original ``get_current_numbers``: parse the ``left:`` set from the
    last line of the trajectory. Returns the original puzzle numbers for the
    empty trajectory (first step).
    """
    last_line = trajectory.strip().split("\n")[-1]
    if "left: " in last_line:
        return last_line.split("left: ")[-1].split(")")[0].strip()
    return last_line.strip()  # step 0: the raw puzzle


class Counter:
    def __init__(self) -> None:
        self.calls = 0


async def _propose(agent, trajectory: str, counter: Counter) -> list[str]:
    """One propose call; returns the raw ``a op b = c (left: ...)`` lines,
    each concatenated onto the running trajectory (as the original does). The
    LM authors its own arithmetic — invalid lines are NOT dropped here.
    """
    counter.calls += 1
    nums = _current_numbers(trajectory)
    text = await agent_runtime.run_agent(agent, PROPOSE_PROMPT.format(input=nums))
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return [trajectory + ln + "\n" for ln in lines]


def _value_name(text: str) -> str:
    """The original ``value_outputs_unwrap``: take the last line of the value
    response and map it to sure/likely/impossible (default impossible)."""
    last = text.strip().split("\n")[-1].strip().lower()
    for name in ("sure", "likely", "impossible"):
        if name in last:
            return name
    return "impossible"


async def _value(agent, trajectory: str, counter: Counter, n: int = 3) -> float:
    """N=3 value samples, SUMMED with the original ad-hoc weights."""
    nums = _current_numbers(trajectory)
    prompt = VALUE_PROMPT.format(input=nums)
    total = 0.0
    for _ in range(n):
        counter.calls += 1
        text = await agent_runtime.run_agent(agent, prompt)
        total += _VALUE_MAP[_value_name(text)]
    return total


async def solve_tot_princeton(
    agent, numbers: tuple[int, ...], beam: int = 5, n_value: int = 3, max_moves: int = 8
) -> tuple[str | None, int]:
    """Original-recipe BFS ToT. Returns ``(winning_expression, llm_calls)``.

    The winning leaf's ``Answer:`` expression IS the answer. Four BFS steps:
    3 intermediate equations + 1 final Answer line (as in the original).
    """
    counter = Counter()
    puzzle_str = " ".join(str(n) for n in numbers)
    beam_states = [puzzle_str + "\n"]

    for step in range(4):  # 3 intermediates + final Answer
        if not beam_states:
            break
        # Propose for every beam state.
        proposals = await asyncio.gather(
            *(_propose(agent, s, counter) for s in beam_states)
        )
        candidates = [p for group in proposals for p in group][: beam * max_moves]
        if not candidates:
            break

        # Value every candidate (N samples, summed).
        values = await asyncio.gather(
            *(_value(agent, c, counter, n_value) for c in candidates)
        )
        # Greedy top-b.
        order = sorted(range(len(candidates)), key=lambda i: values[i], reverse=True)[
            :beam
        ]
        beam_states = [candidates[i] for i in order]

    # Terminal: the best beam's last line should be an Answer line. Extract &
    # verify the expression against the shared oracle.
    for state in beam_states:
        last = state.strip().split("\n")[-1]
        m = re.search(r"answer\s*[:=]\s*(.+)", last, re.IGNORECASE)
        if m:
            expr = m.group(1).strip().rstrip(".").split("=")[0].strip().strip("`")
            ok, _ = check_24(expr, numbers)
            if ok:
                return expr, counter.calls
            # Not 24 — keep looking through the beam.
    # Fallback: try to parse any arithmetic last-line.
    for state in beam_states:
        last = state.strip().split("\n")[-1]
        expr = extract_expression(last)
        ok, _ = check_24(expr, numbers)
        if ok:
            return expr, counter.calls
    return None, counter.calls


# --- Baselines (matched compute) -----------------------------------------


async def solve_single_call(agent, numbers: tuple[int, ...]) -> bool:
    text = await agent_runtime.run_agent(
        agent, COT_PROMPT.format(input=" ".join(str(n) for n in numbers))
    )
    ok, _ = check_24(extract_expression(text), numbers)
    return ok


async def solve_cot_sc(agent, numbers: tuple[int, ...], k: int) -> bool:
    results = await asyncio.gather(
        *(solve_single_call(agent, numbers) for _ in range(k))
    )
    return any(results)


# --- Runner ---------------------------------------------------------------


async def run(limit: int, beam: int, cot_sc_k: int, n_value: int) -> dict:
    agent = create_agent(
        role="Generator", role_name="Solver", model_config=get_model_config("GENERATOR")
    )
    puzzles = PUZZLES[:limit] if limit else PUZZLES

    tot_pass = single_pass = sc_pass = 0
    tot_calls = 0
    rows = []
    for nums in puzzles:
        expr, calls = await solve_tot_princeton(agent, nums, beam=beam, n_value=n_value)
        tot_ok = expr is not None
        single_ok = await solve_single_call(agent, nums)
        sc_ok = await solve_cot_sc(agent, nums, cot_sc_k)
        tot_pass += tot_ok
        single_pass += single_ok
        sc_pass += sc_ok
        tot_calls += calls
        rows.append(
            {
                "puzzle": nums,
                "tot": tot_ok,
                "tot_calls": calls,
                "tot_expr": expr,
                "single_call": single_ok,
                "cot_sc": sc_ok,
            }
        )

    n = len(puzzles)
    return {
        "n": n,
        "beam": beam,
        "n_value": n_value,
        "cot_sc_k": cot_sc_k,
        "tot": tot_pass,
        "single_call": single_pass,
        "cot_sc": sc_pass,
        "avg_tot_calls": round(tot_calls / max(1, n), 1),
        "rows": rows,
    }


def render(result: dict) -> str:
    lines = [
        (
            "# Princeton-recipe ToT vs single call vs CoT-SC | Game-of-24 "
            "(original prompts, LM does arithmetic, ground-truth verified)\n"
        )
    ]
    for r in result["rows"]:
        nums = " ".join(str(x) for x in r["puzzle"])
        lines.append(
            f"[{nums:>14}] tot={int(r['tot'])} ({r['tot_calls']} calls)  "
            f"single={int(r['single_call'])}  cot_sc={int(r['cot_sc'])}  "
            f"{r['tot_expr'] or ''}"
        )
    n = result["n"]
    lines.append(
        f"\n# TOTAL  princeton_ToT={result['tot']}/{n}  "
        f"single_call={result['single_call']}/{n}  "
        f"CoT-SC@{result['cot_sc_k']}={result['cot_sc']}/{n}  "
        f"| avg ToT calls/puzzle={result['avg_tot_calls']}"
    )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Princeton-recipe ToT vs single call on Game-of-24."
    )
    p.add_argument("--limit", type=int, default=0, help="Only run the first N puzzles.")
    p.add_argument("--beam", type=int, default=5, help="BFS beam width b (paper: 5).")
    p.add_argument("--n-value", type=int, default=3, help="Value samples N (paper: 3).")
    p.add_argument("--cot-sc-k", type=int, default=5, help="CoT-SC samples (pass@k).")
    p.add_argument(
        "--json", type=str, default="", help="Write raw results to this path."
    )
    args = p.parse_args()

    result = asyncio.run(run(args.limit, args.beam, args.cot_sc_k, args.n_value))
    print(render(result))
    if args.json:
        import json

        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
