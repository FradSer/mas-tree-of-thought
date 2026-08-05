"""Eval helpers: count LLM calls through the single runtime seam.

LLM calls are counted through ``dialectica.agent_runtime.run_agent``, the same
seam the mocked tests intercept, so cost is measured the way production runs it
— judge calls are deliberately not counted against either contender.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from dialectica import agent_runtime


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
