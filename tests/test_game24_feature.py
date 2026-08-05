"""Step definitions for features/game24.feature — verifier is real, LLM mocked.

The mocked-LLM scenario drives the search with a Python brute-force oracle
(propose = every legal move, value = can-this-reach-24), so it exercises the
full propose -> apply -> value -> BFS -> winning-leaf plumbing and the
ground-truth verifier end to end without a single network call.
"""

import asyncio
import itertools
import operator
from fractions import Fraction
from unittest.mock import patch

from pytest_bdd import given, parsers, scenarios, then, when

from dialectica.agent_factory import create_agent
from evals.game24 import check_24, extract_expression, solve_tot

scenarios("features/game24.feature")


# --- verifier / extraction ------------------------------------------------


@given(parsers.parse('the puzzle "{nums}"'), target_fixture="numbers")
def puzzle(nums: str):
    return tuple(int(x) for x in nums.split())


@when(parsers.parse('the expression "{expr}" is checked'), target_fixture="verdict")
def check_expr(expr: str, numbers):
    return check_24(expr, numbers)


@then("the verification passes")
def verification_passes(verdict):
    assert verdict[0] is True, verdict[1]


@then("the verification fails")
def verification_fails(verdict):
    assert verdict[0] is False


@given("a model answer ending in an Answer line", target_fixture="answer")
def model_answer():
    return (
        "Let me work through it. 6 - 4 = 2, and 4 + 8 = 12, so 12 * 2 = 24.\n"
        "Answer: (4 + 8) * (6 - 4)"
    )


@when("the expression is extracted", target_fixture="extracted")
def do_extract(answer):
    return extract_expression(answer)


@then("it is the arithmetic on that line")
def extracted_matches(extracted):
    assert extracted == "(4 + 8) * (6 - 4)"


# --- faithful ToT end-to-end (oracle-mocked) ------------------------------

_OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}


def _can_make_24(nums: tuple[Fraction, ...]) -> bool:
    """Brute force: can these numbers combine with + - * / to make 24?"""
    if len(nums) == 1:
        return nums[0] == 24
    for i, j in itertools.permutations(range(len(nums)), 2):
        rest = tuple(nums[k] for k in range(len(nums)) if k not in (i, j))
        for sym, fn in _OPS.items():
            if sym == "/" and nums[j] == 0:
                continue
            if _can_make_24(rest + (fn(nums[i], nums[j]),)):
                return True
    return False


def _nums_from_prompt(prompt: str) -> tuple[Fraction, ...]:
    line = next(line for line in prompt.splitlines() if "numbers" in line.lower())
    return tuple(Fraction(tok) for tok in line.split(":")[1].split())


@given(
    "an oracle-mocked LLM that proposes legal moves and values reachable states",
    target_fixture="oracle_llm",
)
def oracle_llm():
    async def fake(agent, instruction: str) -> str:
        nums = _nums_from_prompt(instruction)
        if instruction.lstrip().startswith("Game of 24. You have"):
            moves = []
            for i, j in itertools.permutations(range(len(nums)), 2):
                for sym in _OPS:
                    if sym == "/" and nums[j] == 0:
                        continue
                    moves.append(f"{nums[i]} {sym} {nums[j]} = ?")
            return "\n".join(moves)
        # value prompt
        return "sure" if _can_make_24(nums) else "impossible"

    return fake


@when(
    parsers.parse('faithful ToT runs on the puzzle "{nums}"'),
    target_fixture="tot_result",
)
def run_tot(nums: str, oracle_llm):
    numbers = tuple(int(x) for x in nums.split())
    agent = create_agent(role="Generator", role_name="Solver")
    with patch("dialectica.agent_runtime.run_agent", oracle_llm):
        expr, _calls = asyncio.run(solve_tot(agent, numbers, beam=5))
    return expr, numbers


@then("it returns an expression that the verifier accepts")
def tot_solves(tot_result):
    expr, numbers = tot_result
    assert expr is not None, "ToT failed to find a solution"
    ok, reason = check_24(expr, numbers)
    assert ok, f"{expr!r} rejected: {reason}"
