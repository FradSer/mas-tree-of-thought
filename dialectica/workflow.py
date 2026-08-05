"""Workflow orchestration primitives — a composable multi-agent runtime.

A Python re-implementation of the orchestration surface Claude Code's ``Workflow``
tool provides, built on this repo's single LLM seam (``agent_runtime.run_agent``).
It exists so arbitrary multi-agent workflows — research, review, design,
planning, the *meta-task* regime where generate -> adversarial-judge ->
synthesize genuinely helps — can be expressed as plain Python instead of
hardcoded into one fixed engine loop.

The primitives:

  * ``agent(prompt, *, schema=None, tools=None, instructions="", label=None,
    phase=None, model=None, isolation=None, agent_type=None, sees=None)`` —
    one LLM call. ``isolation="worktree"`` runs in a fresh git worktree;
    ``agent_type`` (e.g. ``"Explore"``) applies a preset charter; ``sees``
    is a per-step access list (Fugu-Ultra-style isolation — see below).
    See module body for full semantics.
  * ``workflow(script_or_name, *, args=None)`` — inline child workflow (one
    nesting level) or registered name via ``register_workflow``.
  * ``run_id()`` — current run id for resume/journaling.
  * ``phase(title)`` / ``log(msg)`` / ``budget()`` / ``args()`` — progress and metering.
  * ``parallel`` / ``pipeline`` — max 4,096 items per call; 1,000 ``agent()``
    calls per run.

HONEST SCOPE: on *self-contained result-quality* tasks, no multi-agent scaffold
in this repo beats a prompt-matched single call (see README Evaluation — the
ToT+GAN engine goes 0-4-1 / 0-2-3 / 0-1-4 vs single / best-of-N / self-refine;
flat self-refine is best). This module is an **orchestration layer for
meta-tasks** (no ground truth, exploratory/judgmental — research, review,
planning, design), NOT a self-contained-quality engine. The existing negative
findings stand; composing a workflow over these primitives does not repeal
them. ``agent(tools=...)`` is the one lever that can: a stage that reads a
real file or runs a real command is grounded the way the agentic pattern is, not a
pure-LLM rearrangement — but only if the caller actually injects tools. A
workflow built entirely from schema-only judge/synthesize stages is still
pure-LLM and still bound by the findings above.

PER-STEP ACCESS LISTS (``sees=``): a direct port of the anti-"orchestration
collapse" mechanism Sakana's Fugu-Ultra uses. In a multi-step workflow the
default is *isolation* — an agent sees only its own prompt, never another
agent's transcript, so the first agent can't silently set the trajectory for
all later ones. ``sees=["gather", "critique"]`` is the opt-in access list:
the outputs of the named prior steps (looked up by their ``label=``) are
appended to this agent's prompt as prior context, and *only* those. This is
the capability-add lever for multi-step independence — it lets a later stage
build on designated earlier outputs without letting every stage see every
prior transcript (which Fugu-Ultra's report shows causes collapse to the
first agent). Unknown labels are ignored, not errors, so an access list
survives conditional branches. Does not require any training (unlike
Fugu-Ultra's learned router); it is a pure context-visibility primitive
over the existing ``parallel``/``pipeline``/``agent`` surface.

Example — a 3-angle research fan-out + synthesis (mocked in tests):

    async def script():
        phase("Gather")
        angles = ["broad", "skeptical", "practitioner"]
        findings = await parallel(
            lambda: agent(f"Research angle: {a}") for a in angles
        )
        phase("Synthesize")
        return await agent(
            "Synthesize: " + " | ".join(f for f in findings if f)
        )

    result = await Workflow(script).run()

Resume / journaling replays the longest unchanged ``agent()`` prefix from a
per-run journal (``.dialectica/workflows/<run_id>/``). Registry-backed
``workflow("name")``, ``meta`` phase validation, lifetime/item caps, and
``agent(isolation="worktree")`` match the Claude Code Workflow tool's
programmatic surface (IDE host UI excluded).
"""

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from . import agent_runtime
from .agent_factory import create_agent
from .agent_runtime import TokenUsage
from .json_repair import repair_json_escapes, strip_code_fence
from .llm_config import _parse_model_config, get_model_config
from .workflow_journal import (
    _MAX_AGENT_CALLS,
    _MAX_ITEM_CAP,
    RunJournal,
    agent_cache_key,
    default_journal_dir,
    deserialize_agent_result,
    serialize_agent_result,
)
from .workflow_registry import get_workflow as _get_registered_workflow
from .workflow_worktree import WorkflowWorktreeError, worktree_path, worktree_session

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Cap overlapping agent() calls. The Workflow tool caps concurrent agent()
# calls (excess calls queue) — the semaphore is acquired inside agent() around
# each underlying LLM call, so parallel/pipeline branches doing non-LLM work
# never hold a slot. The cap is min(16, cpu-2), overridable via
# DIALECTICA_WORKFLOW_CONCURRENCY or the Workflow(concurrency=...) arg.
_DEFAULT_CONCURRENCY = max(1, min(16, (os.cpu_count() or 4) - 2))


class BudgetExhausted(RuntimeError):
    """Raised by ``agent()`` when a budget total is set and exhausted."""


class WorkflowMetaError(ValueError):
    """Raised when ``meta`` is invalid or phases do not match ``phase()`` calls."""


class WorkflowAgentCapExceeded(RuntimeError):
    """Raised when a run exceeds the lifetime agent() cap (1000)."""


@dataclass
class Budget:
    """Call/token budget for a workflow run.

    ``unit`` selects what ``total`` means and what ``spent()``/``remaining()``
    report: ``"calls"`` (default) counts ``agent()`` calls; ``"tokens"``
    counts API-reported output tokens (thinking included — the Workflow
    tool's unit). Both meters always accumulate regardless of unit —
    ``spent_calls()``, ``spent_tokens()`` and ``usage()`` expose them for
    observability even when no total is set. The gate fires at ``agent()``
    entry: once ``spent()`` reaches ``total``, further calls raise
    ``BudgetExhausted`` (a token total can be overshot by the call in flight;
    the next call throws). The pool is per-workflow-run, shared across all
    branches — global, not per-branch. Mocked responses without usage
    metadata (plain strs through the ``run_agent`` seam) meter as zero
    tokens.
    """

    total: int | None = None
    unit: str = "calls"
    _calls: int = field(default=0, init=False, repr=False)
    _prompt_tokens: int = field(default=0, init=False, repr=False)
    _output_tokens: int = field(default=0, init=False, repr=False)
    _total_tokens: int = field(default=0, init=False, repr=False)
    _cached_tokens: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.unit not in ("calls", "tokens"):
            raise ValueError(
                f"unknown budget unit {self.unit!r}: use 'calls' or 'tokens'"
            )

    def spent(self) -> int:
        return self._calls if self.unit == "calls" else self._output_tokens

    def spent_calls(self) -> int:
        return self._calls

    def spent_tokens(self) -> int:
        return self._output_tokens

    def usage(self) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self._prompt_tokens,
            output_tokens=self._output_tokens,
            total_tokens=self._total_tokens,
            cached_tokens=self._cached_tokens,
        )

    def remaining(self) -> float:
        return float("inf") if self.total is None else max(0, self.total - self.spent())

    def _charge(self) -> None:
        if self.total is not None and self.spent() >= self.total:
            noun = "agent calls" if self.unit == "calls" else "output tokens"
            raise BudgetExhausted(
                f"Workflow budget exhausted: {self.spent()}/{self.total} {noun}."
            )
        self._calls += 1

    def _record(self, usage: TokenUsage) -> None:
        self._prompt_tokens += usage.prompt_tokens
        self._output_tokens += usage.output_tokens
        self._total_tokens += usage.total_tokens
        self._cached_tokens += usage.cached_tokens


# --- Run context (ContextVar so nested asyncio tasks see the same context) ---


@dataclass
class _RunCtx:
    budget: Budget
    run_id: str
    journal: RunJournal
    journal_path: Path
    phases: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    concurrency: int = _DEFAULT_CONCURRENCY
    semaphore: asyncio.Semaphore | None = None
    args: Any = None
    depth: int = 0
    agent_sequence: int = 0
    registry_name: str | None = None
    meta: dict[str, Any] | None = None
    # Per-step access-list store: label -> str(result). Populated by agent()
    # after each call so a later agent(sees=[label]) can read its output as
    # prior context. Fugu-Ultra-style isolation: default is "own transcript
    # only"; sees= opts in to designated prior step outputs.
    step_outputs: dict[str, str] = field(default_factory=dict)


_current: ContextVar[_RunCtx | None] = ContextVar(
    "dialectica_workflow_ctx", default=None
)


def _ctx() -> _RunCtx:
    ctx = _current.get()
    if ctx is None:
        raise RuntimeError(
            "Workflow primitives (agent/parallel/pipeline/phase/log/budget) can "
            "only be called inside a Workflow script. See Workflow.run()."
        )
    return ctx


def _to_identifier(label: str) -> str:
    """Coerce a free-form label to a valid Python identifier for LlmAgent.name.

    ADK rejects names containing non-identifier chars (``:`` in
    ``"angle:skeptical"``); the label is for logs, so sanitize it here.
    """
    sanitized = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in label)
    if not sanitized or not sanitized[0].isalpha() and sanitized[0] != "_":
        sanitized = "_" + sanitized
    return sanitized


def _validate_meta(meta: dict[str, Any]) -> None:
    if not meta.get("name") or not meta.get("description"):
        raise WorkflowMetaError("meta requires 'name' and 'description'")


def _check_meta_phases(ctx: _RunCtx) -> None:
    if not ctx.meta:
        return
    declared = [
        p.get("title")
        for p in ctx.meta.get("phases", [])
        if isinstance(p, dict) and p.get("title")
    ]
    if not declared:
        return
    if ctx.phases != declared:
        raise WorkflowMetaError(
            f"meta phases {declared!r} do not match phase() calls {ctx.phases!r}"
        )


def run_id() -> str:
    """The current run's id (for resume/journaling)."""
    return _ctx().run_id


# --- Primitives -----------------------------------------------------------


def phase(title: str) -> None:
    """Mark a progress group (rendering only; no synchronization gate)."""
    ctx = _ctx()
    ctx.phases.append(title)
    logger.info("[workflow phase] %s", title)


def log(message: str) -> None:
    """Emit a progress line to the run log and the module logger."""
    ctx = _ctx()
    ctx.log.append(message)
    logger.info("[workflow] %s", message)


def budget() -> Budget:
    """The run's ``Budget`` — ``spent()`` / ``remaining()`` / ``total``."""
    return _ctx().budget


def args() -> Any:
    """The ``args`` value passed to ``Workflow(...)``, verbatim."""
    return _ctx().args


def in_workflow() -> bool:
    """True when called inside an active ``Workflow`` script context.

    Lets an engine that wraps its own ``Workflow`` script (e.g. repair) join
    an outer run instead of opening a fresh context, so its calls share the
    outer budget and concurrency cap — the Workflow tool's child-workflow rule.
    """
    return _current.get() is not None


async def agent(
    prompt: str,
    *,
    schema: type[BaseModel] | None = None,
    tools: list[Any] | None = None,
    instructions: str = "",
    label: str | None = None,
    phase: str | None = None,
    model: str | None = None,
    max_attempts: int = 3,
    isolation: str | None = None,
    agent_type: str | None = None,
    sees: list[str] | None = None,
) -> Any:
    """One LLM call. Returns the model's text, or a validated ``schema`` instance.

    With ``schema`` set, the agent uses ADK ``output_schema`` for enforced JSON
    and the result is validated into the Pydantic model; a transiently
    empty/malformed response is re-asked up to ``max_attempts`` times, then
    returns ``None`` (a failed agent must not kill its batch). Without ``schema``
    the raw text is returned. ``tools`` wires plain callables (or ADK
    ``FunctionTool``s) into this call's agent so it can act — read a file, run
    a command, query a service — the same wiring the demoted agentic pattern
    (``examples/patterns/agentic_pattern.py``) uses; ADK forbids combining
    ``tools`` with ``schema`` on one ``LlmAgent``, so passing both raises
    ``ValueError`` (run a tool-using stage first, then a separate
    ``schema``-only stage to structure its result). ``instructions`` appends
    task-specific framing to the agent's system prompt (e.g. "act, don't
    guess — verify with tools"), the same role the agentic pattern's system
    prompt plays, without needing a separate engine class. ``model`` overrides the
    model for this call only (``"openai:qwen3.6-flash"`` style, resolved the
    same way every other roster call site in this repo resolves it; ``None``
    inherits the session default). ``sees`` is a per-step access list of prior
    step labels (``sees=["gather", "critique"]``): the *outputs* of those
    prior steps (looked up by their ``label=``) are appended to this call's
    prompt as prior context, and only those — the default (``None``) is full
    isolation, an agent never sees another agent's transcript. Unknown labels
    are ignored, so an access list survives conditional branches. This is the
    Fugu-Ultra anti-collapse lever (see module docstring): selective context
    visibility, opt-in, no training required.
    """
    if tools and schema is not None:
        raise ValueError(
            "agent() cannot combine tools with schema — ADK forbids tools + "
            "output_schema on one LlmAgent. Run a tool-using stage (no schema) "
            "first, then a separate schema-only stage to structure its result."
        )
    if isolation is not None and isolation != "worktree":
        raise ValueError(
            f"unknown isolation {isolation!r}: only 'worktree' is supported"
        )

    ctx = _ctx()
    if ctx.agent_sequence >= _MAX_AGENT_CALLS:
        raise WorkflowAgentCapExceeded(
            f"Workflow agent cap exceeded: {_MAX_AGENT_CALLS} agent() calls per run."
        )

    cache_key = agent_cache_key(
        prompt,
        schema=schema,
        tools=tools,
        instructions=instructions,
        label=label,
        phase=phase,
        model=model,
        isolation=isolation,
        agent_type=agent_type,
        sees=sees,
    )
    sequence = ctx.agent_sequence
    cached = ctx.journal.lookup(sequence, cache_key)
    if cached is not None:
        ctx.agent_sequence += 1
        return deserialize_agent_result(cached, schema)

    ctx.budget._charge()
    if phase:
        ctx.phases.append(phase)

    worktree_note = ""
    if isolation == "worktree":
        worktree_note = (
            "\n\nYou are running in an isolated git worktree. "
            "Confine file edits and commands to the current working directory."
        )

    name = _to_identifier(label) if label else "WorkflowAgent"

    async def _execute_agent() -> Any:
        agent_obj = create_agent(
            role="Generator",
            role_name=name,
            additional_context=instructions + worktree_note,
            model_config=_parse_model_config(model)
            if model
            else get_model_config("GENERATOR"),
            tools=tools,
            output_schema=schema,
            agent_type=agent_type,
        )

        effective_prompt = prompt
        if sees:
            prior = ctx.step_outputs
            # Only steps that have *already finished* contribute; a label not
            # yet written (a concurrent parallel() sibling that hasn't
            # completed, or one that never ran) is skipped, not an error.
            # This keeps access lists resilient to conditional branches and
            # avoids a partial-read race under concurrency: the write happens
            # after this agent's own LLM call returns, so a sees= reference
            # to a not-yet-finished sibling is treated as absent rather than
            # reading a half-written value. Callers wanting a guaranteed
            # visible dependency should run the producer before the consumer
            # (await it sequentially, not as a parallel() sibling).
            blocks = [prior[s] for s in sees if s in prior]
            if blocks:
                effective_prompt = (
                    "Prior context from named steps:\n"
                    + "\n---\n".join(blocks)
                    + "\n---\n\nTask:\n"
                    + prompt
                )
        # The JSON-format instruction is gated on the *task* prompt (not the
        # prior-context-augmented effective_prompt): an injected prior output
        # that happens to mention "json" must not suppress the schema
        # formatting instruction and risk prose -> parse failure.
        if schema is not None and "json" not in prompt.lower():
            effective_prompt = (
                effective_prompt + "\n\nReturn your answer as a single JSON object."
            )

        sem = ctx.semaphore

        async def _call() -> str:
            if sem is None:
                response = await agent_runtime.run_agent(agent_obj, effective_prompt)
            else:
                async with sem:
                    response = await agent_runtime.run_agent(
                        agent_obj, effective_prompt
                    )
            usage = getattr(response, "usage", None)
            if usage is not None:
                ctx.budget._record(usage)
            return response

        response = await _call()

        if schema is None:
            return response

        for attempt in range(1, max_attempts + 1):
            result = _parse_structured(response, schema)
            if result is not None:
                return result
            if attempt < max_attempts:
                logger.warning(
                    "agent(%s) unparseable structured output, re-asking %d/%d",
                    label or "agent",
                    attempt + 1,
                    max_attempts,
                )
                response = await _call()
        logger.warning(
            "agent(%s) returning None after %d parse failures", label, max_attempts
        )
        return None

    if isolation == "worktree":
        async with worktree_session(label=name, run_id=ctx.run_id) as handle:
            result = await _execute_agent()
            if handle.dirty:
                log(f"worktree kept (dirty): {handle.path}")
    else:
        result = await _execute_agent()

    entry = serialize_agent_result(result, schema)
    entry.sequence = sequence
    entry.cache_key = cache_key
    entry.prompt = prompt
    ctx.journal.append(entry)
    # Record this step's output (by label) so a later agent(sees=[label])
    # can read it as prior context. Pydantic models are serialized to their
    # JSON form — the same representation the journal stores — so a schema
    # step is consumable the same way a text step is.
    if label is not None and result is not None:
        if isinstance(result, BaseModel):
            ctx.step_outputs[label] = result.model_dump_json()
        else:
            ctx.step_outputs[label] = str(result)
    ctx.agent_sequence += 1
    ctx.journal.persist(ctx.journal_path)
    return result


def _parse_structured(response: str, schema: type[BaseModel]) -> BaseModel | None:
    """Parse an LLM JSON response into ``schema``, tolerating fences/escapes.

    Models in schema mode often wrap the JSON object in narration or a markdown
    fence ("Here is the result: ```json {...} ```"). ``strip_code_fence`` only
    handles the case where the WHOLE body is fenced, so we additionally extract
    the first balanced ``{...}`` object and try every candidate.
    """
    if not response:
        return None
    candidates = _extract_json_candidates(response)
    for body in candidates:
        for candidate in (body, repair_json_escapes(body)):
            try:
                return schema.model_validate_json(candidate)
            except ValidationError:
                continue
            except ValueError:
                continue
    return None


def _extract_json_candidates(response: str) -> list[str]:
    """Yield plausible JSON-object substrings to try, most specific first.

    Order: the fenced body if any, then the first balanced ``{...}`` span,
    then the stripped whole response as a last resort.
    """
    candidates: list[str] = []
    fenced = strip_code_fence(response).strip()
    if fenced:
        candidates.append(fenced)
    span = _first_json_object_span(response)
    if span is not None:
        candidates.append(response[span[0] : span[1]])
    candidates.append(response.strip())
    # Dedup while preserving order.
    seen: set[str] = set()
    return [c for c in candidates if not (c in seen or seen.add(c))]


def _first_json_object_span(text: str) -> tuple[int, int] | None:
    """Return (start, end) indices of the first balanced top-level ``{...}``."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return (start, i + 1)
        start = text.find("{", start + 1)
    return None


async def parallel(thunks: Sequence[Callable[[], Awaitable[Any]]]) -> list[Any]:
    """Run zero-arg coroutines concurrently and WAIT for all (a barrier).

    A thunk that raises (or whose awaited value raises) resolves to ``None``;
    the call never rejects. Filter with ``[x for x in res if x is not None]``.
    Thunks hold no concurrency slot themselves — the run's cap gates the
    ``agent()`` calls inside them.
    """
    _ctx()  # primitives are only valid inside a Workflow script
    if len(thunks) > _MAX_ITEM_CAP:
        raise ValueError(
            f"parallel() accepts at most {_MAX_ITEM_CAP} thunks, got {len(thunks)}"
        )

    async def _run(thunk: Callable[[], Awaitable[Any]]) -> Any:
        try:
            return await thunk()
        except Exception as e:  # noqa: BLE001 —Workflow contract: failure -> null
            logger.warning("parallel thunk failed: %s", e)
            return None

    return await asyncio.gather(*(_run(t) for t in thunks))


async def pipeline(
    items: Sequence[Any], *stages: Callable[[Any, Any, int], Awaitable[Any]]
) -> list[Any]:
    """Run each item through every stage with NO barrier between stages.

    Item A can be in stage 3 while item B is still in stage 1 — wall-clock is
    the slowest single-item chain, not the sum of slowest-per-stage. Each stage
    callback receives ``(prev_result, original_item, index)``; use
    ``original_item``/``index`` to label work without threading context through
    stage 1's return value. A stage that throws drops that item to ``None`` and
    skips its remaining stages. The run's concurrency cap gates the ``agent()``
    calls inside stages, not whole item chains — an item idling in a non-LLM
    stage holds no slot.
    """
    _ctx()  # primitives are only valid inside a Workflow script
    if len(items) > _MAX_ITEM_CAP:
        raise ValueError(
            f"pipeline() accepts at most {_MAX_ITEM_CAP} items, got {len(items)}"
        )

    async def _chain(item: Any, index: int) -> Any:
        try:
            prev = item
            for stage in stages:
                prev = await stage(prev, item, index)
            return prev
        except Exception as e:  # noqa: BLE001 — pipeline contract: failure -> null
            logger.warning("pipeline item %d dropped: %s", index, e)
            return None

    return await asyncio.gather(*(_chain(it, i) for i, it in enumerate(items)))


async def workflow(
    script_or_name: Callable[[], Awaitable[Any]] | str,
    *,
    args: Any = None,
) -> Any:
    """Run a workflow script, joining the parent run when already inside one.

    ``script_or_name`` may be a coroutine function or a registered workflow name.
    Standalone opens a fresh run; inside an outer script at depth 0 joins that
    run's budget and concurrency cap. Nesting is limited to one level.
    """
    if isinstance(script_or_name, str):
        script = _get_registered_workflow(script_or_name)
        registry_name = script_or_name
    else:
        script = script_or_name
        registry_name = None

    ctx = _current.get()
    if ctx is None:
        return await Workflow(script, args=args, registry_name=registry_name).run()
    if ctx.depth >= 1:
        raise RuntimeError(
            "workflow() nesting is limited to one level — workflow() cannot "
            "be called inside a child workflow."
        )
    prev_args = ctx.args
    prev_registry = ctx.registry_name
    ctx.depth = 1
    ctx.args = args
    ctx.registry_name = registry_name
    try:
        return await script()
    finally:
        ctx.depth = 0
        ctx.args = prev_args
        ctx.registry_name = prev_registry


# --- Entry point -----------------------------------------------------------


class Workflow:
    """Executes a workflow script with the primitives in scope.

    The script is a zero-arg coroutine that calls ``agent``/``parallel``/
    ``pipeline``/``phase``/``log``/``budget``/``args`` as module-level names —
    they resolve to the current run's context via ``ContextVar``, so nested
    ``asyncio`` tasks (created by ``parallel``/``pipeline``) see the same
    budget and log.
    """

    def __init__(
        self,
        script: Callable[[], Awaitable[Any]],
        *,
        args: Any = None,
        budget_total: int | None = None,
        budget_unit: str = "calls",
        concurrency: int | None = None,
        meta: dict[str, Any] | None = None,
        resume_run_id: str | None = None,
        journal_dir: str | Path | None = None,
        registry_name: str | None = None,
    ):
        self.script = script
        self._args = args
        self._budget_total = budget_total
        self._budget_unit = budget_unit
        self._concurrency = concurrency
        self._meta = meta
        self._resume_run_id = resume_run_id
        self._journal_dir = Path(journal_dir) if journal_dir else default_journal_dir()
        self._registry_name = registry_name

    async def run(self) -> Any:
        if self._meta is not None:
            _validate_meta(self._meta)
        cap = (
            self._concurrency
            or int(os.environ.get("DIALECTICA_WORKFLOW_CONCURRENCY", "0") or "0")
            or _DEFAULT_CONCURRENCY
        )
        journal, journal_path = RunJournal.create(
            script=self.script,
            args=self._args,
            registry_name=self._registry_name,
            resume_run_id=self._resume_run_id,
            journal_dir=self._journal_dir,
        )
        ctx = _RunCtx(
            budget=Budget(total=self._budget_total, unit=self._budget_unit),
            run_id=journal.run_id,
            journal=journal,
            journal_path=journal_path,
            concurrency=cap,
            semaphore=asyncio.Semaphore(cap),
            args=self._args,
            registry_name=self._registry_name,
            meta=self._meta,
        )
        token = _current.set(ctx)
        try:
            result = await self.script()
            _check_meta_phases(ctx)
            return result
        finally:
            ctx.journal.persist(ctx.journal_path)
            _current.reset(token)


__all__ = [
    "Budget",
    "BudgetExhausted",
    "Workflow",
    "WorkflowAgentCapExceeded",
    "WorkflowMetaError",
    "WorkflowWorktreeError",
    "agent",
    "args",
    "budget",
    "in_workflow",
    "log",
    "parallel",
    "phase",
    "pipeline",
    "run_id",
    "workflow",
    "worktree_path",
]

from .workflow_registry import list_workflows, register_workflow

__all__ += ["list_workflows", "register_workflow"]
