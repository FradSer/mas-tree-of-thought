"""Agentic engine vs single call — the regime where an engine genuinely wins.

Self-contained evals showed no scaffold beats a well-prompted single call (the
model does the task in one pass). An engine only wins when it adds capability a
single forward pass CANNOT have: acting on the world and iterating. This eval
demonstrates that cleanly and objectively.

Task: discover a hidden int->int rule, then implement a matching `g(x)`. The
agent may call probe(x) to observe the hidden function and infer the rule; the
single call has no probe, so it must guess an arbitrary rule and fails. `g` is
checked on secret held-out inputs in a subprocess (objective, no LLM judge).

Live tool (needs a model via OPENAI_API_BASE/_KEY + DEFAULT_/GENERATOR_MODEL_CONFIG).
Run:  uv run python -m evals.agentic_eval
"""

import argparse
import asyncio
import subprocess
import sys

from dialectica import agent_runtime
from dialectica.agent_factory import create_agent
from dialectica.llm_config import get_model_config
from evals.code_eval import extract_python_code
from examples.patterns.agentic_pattern import create_agentic_engine

# Arbitrary-but-learnable hidden rules: not guessable from the bare spec,
# inferable from a handful of probes.
ORACLES = {
    "linmod": lambda x: (x * 7 + 13) % 100,
    "quad": lambda x: x * x - x,
    "digitsum3": lambda x: sum(int(d) for d in str(abs(x))) * 3,
    "collatz-step": lambda x: x // 2 if x % 2 == 0 else 3 * x + 1,
    "bitmix": lambda x: (x << 2) ^ 5,
    "popcount": lambda x: abs(x).bit_count(),
    "reverse": lambda x: (-1 if x < 0 else 1) * int(str(abs(x))[::-1]),
    "relu": lambda x: max(0, x * 2 - 7),
}
HELDOUT = [3, 17, 42, 100, 7, 256, 9, 1000]  # secret test inputs

AGENT_TASK = (
    "There is a hidden function mapping one integer to one integer by a fixed but "
    "unknown rule. Discover the rule, then implement it as a Python function `g(x)`.\n"
    "Call the probe tool with any integer to see the hidden function's output — probe "
    "as many inputs as you need to be confident.\n"
    "When sure, return ONLY the implementation of `g` in a single ```python code "
    "block. `g` must be self-contained and must NOT call probe."
)
BASELINE_TASK = (
    "There is a hidden function mapping one integer to one integer by a fixed but "
    "unknown rule. Implement your best guess as a Python function `g(x)`. Return ONLY "
    "the implementation in a single ```python code block."
)


def test_g(code: str, f) -> bool:
    asserts = "\n".join(f"assert g({t}) == {f(t)}" for t in HELDOUT)
    try:
        r = subprocess.run(
            [sys.executable, "-c", f"{code}\n\n{asserts}\n"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def make_probe(f, counter):
    def probe(x: int) -> int:
        """Return the hidden function's output for the integer x."""
        counter[0] += 1
        return f(x)

    return probe


async def agent_arm(f):
    counter = [0]
    engine = create_agentic_engine(
        AGENT_TASK,
        tools=[make_probe(f, counter)],
        model_config=get_model_config("GENERATOR"),
    )
    out = (await engine.run())["final_answer"]
    return test_g(extract_python_code(out), f), counter[0]


async def baseline_arm(f):
    agent = create_agent(
        role="Generator", role_name="Solver", model_config=get_model_config("GENERATOR")
    )
    out = await agent_runtime.run_agent(agent, BASELINE_TASK)
    return test_g(extract_python_code(out), f)


async def run() -> str:
    lines = ["# agentic (probe tool) vs single call | hidden-oracle discovery\n"]
    a_pass = b_pass = probes = 0
    for name, f in ORACLES.items():
        ap, n = await agent_arm(f)
        bp = await baseline_arm(f)
        a_pass += ap
        b_pass += bp
        probes += n
        lines.append(f"{name}: agent={int(ap)} (probes={n})  single_call={int(bp)}")
    n_total = len(ORACLES)
    lines.append(
        f"\n# TOTAL  agent={a_pass}/{n_total}  single_call={b_pass}/{n_total}  "
        f"| avg probes/task={probes / max(1, n_total):.1f}"
    )
    return "\n".join(lines)


def main() -> None:
    argparse.ArgumentParser(
        description="Agentic vs single-call oracle eval."
    ).parse_args()
    print(asyncio.run(run()))


if __name__ == "__main__":
    main()
