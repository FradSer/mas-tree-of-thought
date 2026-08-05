"""Single entry point for invoking an LlmAgent.

Centralizing the ADK Runner call gives every pluggable component (generator,
evaluator, synthesizer) one shared seam — which is also the one place tests
patch to run the engine without the network.

``run_agent`` retries transient failures with exponential backoff: an engine
run is hundreds of sequential LLM calls, and without retry a single network
error or rate limit throws the whole run away. Persistent failures re-raise.
"""

import asyncio
import logging
import os
import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self

from google.adk.agents import LlmAgent
from google.adk.apps.app import App
from google.adk.events import Event
from google.adk.runners import InMemoryRunner

from .adk_config import (
    ensure_otel_setup,
    get_context_cache_config,
    reset_adk_config_state,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenUsage:
    """API-reported token counts for one or more LLM calls.

    ``output_tokens`` includes thinking tokens (billed as output). All zeros
    when the backend reports no usage metadata.
    """

    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0


class AgentResponse(str):
    """The agent's text output, carrying the call's ``TokenUsage``.

    A ``str`` subclass so every consumer of the ``run_agent`` seam — and every
    test fake that returns a plain str — keeps working unchanged; metering
    callers read ``.usage``.
    """

    usage: TokenUsage

    def __new__(cls, text: str, usage: TokenUsage) -> "Self":
        obj = super().__new__(cls, text)
        obj.usage = usage
        return obj

    def __getnewargs__(self) -> tuple:  # str's default drops ``usage`` on pickle
        return (str(self), self.usage)


def _usage_from_events(events: Iterable[Event]) -> TokenUsage:
    """Sum ``usage_metadata`` across a run's events.

    One event per LLM turn — a tool-using call produces several. Events
    without metadata (e.g. function-call events) count as zero.

    Output tokens: native Gemini reports ``candidates_token_count`` EXCLUDING
    thoughts, but ADK's LiteLLM mapping sets it to ``completion_tokens`` —
    which already INCLUDES reasoning — and reports ``thoughts_token_count`` on
    top. Naively summing both double-counts reasoning on ``openai:`` roster
    models, so when a total is reported, thoughts are clamped to the room the
    totals actually leave (``total - prompt - candidates``); Gemini's totals
    leave exactly ``thoughts``, LiteLLM's leave zero.
    """
    prompt = output = total = cached = 0
    for event in events:
        um = event.usage_metadata
        if um is None:
            continue
        event_prompt = um.prompt_token_count or 0
        candidates = um.candidates_token_count or 0
        thoughts = um.thoughts_token_count or 0
        event_total = um.total_token_count or 0
        if event_total and thoughts:
            thoughts = min(thoughts, max(0, event_total - event_prompt - candidates))
        prompt += event_prompt
        output += candidates + thoughts
        total += event_total
        cached += um.cached_content_token_count or 0
    return TokenUsage(
        prompt_tokens=prompt,
        output_tokens=output,
        total_tokens=total,
        cached_tokens=cached,
    )


# Optional global cap on concurrent LLM calls, for tightly-quota'd backends
# (e.g. gemma-4-31b allows only 16k input tokens/minute — unbounded gather
# self-collides on the quota). 0 or unset = unlimited.
_concurrency_limiter: asyncio.Semaphore | None = None
_limiter_configured = False


def _reset_concurrency_limiter() -> None:
    """Re-read DIALECTICA_MAX_CONCURRENCY on next call (used by tests)."""
    global _concurrency_limiter, _limiter_configured
    _concurrency_limiter = None
    _limiter_configured = False


def _get_concurrency_limiter() -> asyncio.Semaphore | None:
    global _concurrency_limiter, _limiter_configured
    if not _limiter_configured:
        _limiter_configured = True
        cap = int(os.environ.get("DIALECTICA_MAX_CONCURRENCY", "0") or "0")
        if cap > 0:
            _concurrency_limiter = asyncio.Semaphore(cap)
    return _concurrency_limiter


def _make_runner(agent: LlmAgent) -> InMemoryRunner:
    """Build an ADK runner, optionally wiring ADK 2.3 context caching via ``App``."""
    cache_config = get_context_cache_config()
    if cache_config is not None:
        app = App(
            name="dialectica",
            root_agent=agent,
            context_cache_config=cache_config,
        )
        return InMemoryRunner(app=app, app_name="dialectica")
    return InMemoryRunner(agent=agent, app_name="dialectica")


def _reset_adk_runtime_state() -> None:
    """Re-read ADK env config on next call (tests only)."""
    reset_adk_config_state()


async def _call_agent_once(agent: LlmAgent, instruction: str) -> str:
    """One raw ADK invocation, returning the concatenated text output.

    The return value is an ``AgentResponse`` — a str carrying the call's
    summed ``TokenUsage`` for metering callers.
    """
    ensure_otel_setup()
    runner = _make_runner(agent)
    events = await runner.run_debug(instruction, quiet=True)

    response_text = ""
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text and not part.thought:
                    response_text += part.text

    return AgentResponse(response_text.strip(), _usage_from_events(events))


# Rate-limit quotas (e.g. tokens-per-minute) need the window to roll over;
# exponential backoff in seconds just burns the remaining attempts.
RATE_LIMIT_COOLDOWN = 45.0
MAX_RATE_LIMIT_RETRIES = 8

_RATE_LIMIT_MARKERS = ("429", "RESOURCE_EXHAUSTED", "rate limit", "RateLimit")


def _is_rate_limited(error: Exception) -> bool:
    text = str(error)
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


async def run_agent(
    agent: LlmAgent,
    instruction: str,
    *,
    max_attempts: int = 3,
    base_delay: float = 2.0,
) -> str:
    """Run ``agent`` on ``instruction``, retrying transient failures.

    Rate-limit errors (429/RESOURCE_EXHAUSTED) have their own retry budget
    (``MAX_RATE_LIMIT_RETRIES``) and wait ``RATE_LIMIT_COOLDOWN`` (scaled,
    jittered to desynchronize concurrent callers) so the quota window can
    roll over; other failures use ``max_attempts`` with fast exponential
    backoff. ``DIALECTICA_MAX_CONCURRENCY`` caps overlapping calls globally.
    From the real transport the returned str is an ``AgentResponse`` whose
    ``.usage`` carries API-reported token counts; a plain str (e.g. from a
    test fake) simply meters as zero. Usage covers the returned attempt only —
    an attempt that raises partway (even after billable turns) discards its
    events, so those tokens are not metered.
    """
    limiter = _get_concurrency_limiter()
    failures = 0
    rate_limit_hits = 0
    while True:
        try:
            if limiter is not None:
                async with limiter:
                    return await _call_agent_once(agent, instruction)
            return await _call_agent_once(agent, instruction)
        except Exception as e:
            if _is_rate_limited(e):
                rate_limit_hits += 1
                if rate_limit_hits > MAX_RATE_LIMIT_RETRIES:
                    raise
                delay = RATE_LIMIT_COOLDOWN * min(rate_limit_hits, 3)
                delay += random.uniform(0, 10)
                logger.warning(
                    "Rate limited (hit %d/%d), cooling down %.1fs",
                    rate_limit_hits,
                    MAX_RATE_LIMIT_RETRIES,
                    delay,
                )
            else:
                failures += 1
                if failures >= max_attempts:
                    raise
                delay = base_delay * 2 ** (failures - 1)
                logger.warning(
                    "Agent call failed (attempt %d/%d), retrying in %.1fs: %s",
                    failures,
                    max_attempts,
                    delay,
                    e,
                )
            await asyncio.sleep(delay)
