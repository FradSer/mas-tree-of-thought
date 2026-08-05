"""Step definitions for features/workflow.feature — LLM fully mocked.

The primitives are driven through the same ``agent_runtime.run_agent`` seam the
rest of the suite mocks, so these tests exercise the real orchestration logic
(parallel barrier, pipeline no-barrier, schema validation, budget gate) with a
deterministic fake LLM.
"""

import asyncio
from unittest.mock import patch

import pytest
from pydantic import BaseModel
from pytest_bdd import given, scenarios, then, when

from dialectica import workflow as wf
from dialectica.agent_runtime import AgentResponse, TokenUsage
from dialectica.workflow import Budget, BudgetExhausted, Workflow

scenarios("features/workflow.feature")


# --- shared fakes ---------------------------------------------------------


class _Item(BaseModel):
    label: str
    score: int


def _call_counter():
    state = {"calls": 0, "responses": []}

    async def fake(agent, instruction: str) -> str:
        state["calls"] += 1
        return state["responses"].pop(0) if state["responses"] else "ok"

    return fake, state


# --- Scenario 1: parallel barrier + failure -> null ---------------------


@given(
    'a mocked LLM that returns a string per call and fails on "bad"',
    target_fixture="parallel_llm",
)
def parallel_llm():
    async def fake(agent, instruction: str) -> str:
        if "bad" in instruction:
            raise RuntimeError("boom")
        return instruction.replace("call:", "")

    return fake


@when("parallel runs three thunks including a failing one", target_fixture="result")
def run_parallel(parallel_llm):
    async def script():
        return await wf.parallel(
            [
                lambda: wf.agent("call:alpha"),
                lambda: wf.agent("call:bad"),
                lambda: wf.agent("call:gamma"),
            ]
        )

    with patch("dialectica.agent_runtime.run_agent", parallel_llm):
        return asyncio.run(Workflow(script).run())


@then("it returns three results with the failed one null")
def parallel_has_null(result):
    assert len(result) == 3
    assert result[1] is None


@then("the other two are their strings")
def parallel_survivors(result):
    assert result[0] == "alpha"
    assert result[2] == "gamma"


# --- Scenario 2: pipeline no-barrier --------------------------------------


@given("a mocked LLM and an increment stage", target_fixture="pipeline_setup")
def pipeline_setup():
    async def fake(agent, instruction: str) -> str:
        return instruction  # echo; stages do the real work

    def stage_upper(prev, item, index):
        async def _do():
            return await wf.agent(f"{item}-{index}")

        return _do()

    return fake, stage_upper


@when("pipeline runs three items through two stages", target_fixture="pipeline_result")
def run_pipeline(pipeline_setup):
    fake, stage_a = pipeline_setup

    async def stage_b(prev, item, index):
        return f"[{prev}]"

    async def script():
        return await wf.pipeline(["x", "y", "z"], stage_a, stage_b)

    with patch("dialectica.agent_runtime.run_agent", fake):
        return asyncio.run(Workflow(script).run())


@then("each item reaches the final stage independently")
def pipeline_final(pipeline_result):
    assert pipeline_result == ["[x-0]", "[y-1]", "[z-2]"]


# --- Scenario 3: pipeline stage throws -> item null -----------------------


@given(
    "a mocked LLM and a stage that throws on item index 1",
    target_fixture="throw_setup",
)
def throw_setup():
    async def fake(agent, instruction: str) -> str:
        return "ok"

    async def stage(prev, item, index):
        if index == 1:
            raise RuntimeError("drop this one")
        return f"{item}-{index}"

    return fake, stage


@when("pipeline runs three items", target_fixture="throw_result")
def run_pipeline_throw(throw_setup):
    fake, stage = throw_setup

    async def script():
        return await wf.pipeline(["a", "b", "c"], stage)

    with patch("dialectica.agent_runtime.run_agent", fake):
        return asyncio.run(Workflow(script).run())


@then("item 0 and item 2 survive and item 1 is null")
def pipeline_drops_one(throw_result):
    assert throw_result == ["a-0", None, "c-2"]


# --- Scenario 4: agent with schema -> validated instance -------------------


@given("a mocked LLM that returns valid JSON for a schema", target_fixture="schema_llm")
def schema_llm():
    async def fake(agent, instruction: str) -> str:
        return '{"label": "thing", "score": 7}'

    return fake


@when("agent runs with the schema", target_fixture="schema_result")
def run_agent_schema(schema_llm):
    async def script():
        return await wf.agent("judge this", schema=_Item, label="judge")

    with patch("dialectica.agent_runtime.run_agent", schema_llm):
        return asyncio.run(Workflow(script).run())


@then("it returns a validated instance with the fields")
def agent_validated(schema_result):
    assert isinstance(schema_result, _Item)
    assert schema_result.label == "thing"
    assert schema_result.score == 7


# --- Scenario 5: agent without schema -> raw text -------------------------


@given("a mocked LLM that returns prose", target_fixture="prose_llm")
def prose_llm():
    async def fake(agent, instruction: str) -> str:
        return "just some text"

    return fake


@when("agent runs without a schema", target_fixture="prose_result")
def run_agent_prose(prose_llm):
    async def script():
        return await wf.agent("hello")

    with patch("dialectica.agent_runtime.run_agent", prose_llm):
        return asyncio.run(Workflow(script).run())


@then("it returns the raw text")
def agent_raw(prose_result):
    assert prose_result == "just some text"


# --- Scenario 6: agent with schema -> null after parse failures -----------


@given("a mocked LLM that always returns unparseable JSON", target_fixture="bad_llm")
def bad_llm():
    async def fake(agent, instruction: str) -> str:
        return "this is not json at all"

    return fake


@when("agent retries a schema on unparseable output", target_fixture="bad_result")
def run_agent_bad(bad_llm):
    async def script():
        return await wf.agent("judge", schema=_Item, label="judge", max_attempts=2)

    with patch("dialectica.agent_runtime.run_agent", bad_llm):
        return asyncio.run(Workflow(script).run())


@then("it returns null")
def agent_null(bad_result):
    assert bad_result is None


# --- Scenario 6b: fenced/narrated JSON parses ----------------------------


@given(
    "a mocked LLM that returns a fenced JSON object inside prose",
    target_fixture="fenced_llm",
)
def fenced_llm():
    async def fake(agent, instruction: str) -> str:
        return 'Here is the result:\n```json\n{"label": "thing", "score": 9}\n```'

    return fake


@when(
    "agent parses the fenced response with the schema", target_fixture="fenced_result"
)
def run_agent_fenced(fenced_llm):
    async def script():
        return await wf.agent("judge this", schema=_Item, label="judge")

    with patch("dialectica.agent_runtime.run_agent", fenced_llm):
        return asyncio.run(Workflow(script).run())


@then("the fenced response yields a validated instance")
def fenced_validated(fenced_result):
    assert isinstance(fenced_result, _Item)
    assert fenced_result.label == "thing"
    assert fenced_result.score == 9


# --- Scenario 7: budget gate ---------------------------------------------


@given("a mocked LLM and a budget of one call", target_fixture="budget_setup")
def budget_setup():
    async def fake(agent, instruction: str) -> str:
        return "ok"

    return fake


@when(
    "agent runs a second time after the first consumes the budget",
    target_fixture="budget_result",
)
def run_budget(budget_setup):
    fake = budget_setup
    state = {"second": None, "raised": None}

    async def script():
        await wf.agent("first")
        try:
            await wf.agent("second")
        except BudgetExhausted as e:
            state["raised"] = str(e)
        return state

    with patch("dialectica.agent_runtime.run_agent", fake):
        return asyncio.run(Workflow(script, budget_total=1).run())


@then("it raises BudgetExhausted")
def budget_raised(budget_result):
    assert budget_result["raised"] is not None
    assert "1/1" in budget_result["raised"]


# --- Scenario 8: phase/log captured --------------------------------------


@given("a workflow script that phases and logs", target_fixture="phase_script")
def phase_script():
    # Return the script AND a holder; the script writes the run context (which
    # is only valid mid-run) into the holder before run() tears it down.
    holder = {"ctx": None}

    async def script():
        wf.phase("Gather")
        wf.log("starting")
        await wf.agent("x")
        wf.phase("Done")
        wf.log("finished")
        holder["ctx"] = wf._current.get()  # snapshot before run() returns
        return holder

    return script


@when("the workflow runs", target_fixture="phase_result")
def run_phase(phase_script):
    async def fake(agent, instruction: str) -> str:
        return "ok"

    with patch("dialectica.agent_runtime.run_agent", fake):
        return asyncio.run(Workflow(phase_script).run())


@then("the phases and log are captured on the run")
def phase_captured(phase_result):
    ctx = phase_result["ctx"]
    assert ctx is not None
    assert ctx.phases == ["Gather", "Done"]
    assert ctx.log == ["starting", "finished"]
    assert ctx.budget.spent() == 1


# --- Scenario 9: agent wires injected tools into the underlying agent -----


def _noop_tool(x: str) -> str:
    """A trivial tool for wiring assertions."""
    return x


@given("a mocked LLM that records the agent it receives", target_fixture="capture_llm")
def capture_llm():
    captured = {"agent": None}

    async def fake(agent, instruction: str) -> str:
        captured["agent"] = agent
        return "ok"

    return fake, captured


@when("agent runs with a tool injected", target_fixture="tools_result")
def run_agent_tools(capture_llm):
    fake, captured = capture_llm

    async def script():
        return await wf.agent("do it", tools=[_noop_tool], label="doer")

    with patch("dialectica.agent_runtime.run_agent", fake):
        asyncio.run(Workflow(script).run())
    return captured


@then("the underlying agent carries that tool")
def tools_wired(tools_result):
    assert _noop_tool in tools_result["agent"].tools


# --- Scenario 10: tools + schema is rejected ------------------------------


@when("agent runs with both tools and a schema", target_fixture="conflict_error")
def run_agent_conflict():
    async def fake(agent, instruction: str) -> str:
        return "ok"

    async def script():
        await wf.agent("do it", tools=[_noop_tool], schema=_Item)

    with patch("dialectica.agent_runtime.run_agent", fake):
        try:
            asyncio.run(Workflow(script).run())
            return None
        except ValueError as e:
            return e


@then("it raises ValueError naming the ADK conflict")
def conflict_raised(conflict_error):
    assert conflict_error is not None
    assert "tools" in str(conflict_error) and "schema" in str(conflict_error)


# --- Scenario 11: agent(instructions=...) reaches the system prompt -------


@when("agent runs with instructions injected", target_fixture="instructions_result")
def run_agent_instructions(capture_llm):
    fake, captured = capture_llm

    async def script():
        return await wf.agent(
            "do it", instructions="Verify with tools before answering.", label="doer"
        )

    with patch("dialectica.agent_runtime.run_agent", fake):
        asyncio.run(Workflow(script).run())
    return captured


@then("the underlying agent's instruction contains the injected text")
def instructions_wired(instructions_result):
    assert (
        "Verify with tools before answering."
        in instructions_result["agent"].instruction
    )


# --- Scenario 12: agent(model=...) resolves provider:model before build ---


@when(
    "agent runs with a provider-prefixed model override", target_fixture="model_result"
)
def run_agent_model_override(capture_llm):
    fake, captured = capture_llm

    async def script():
        return await wf.agent("do it", model="google:gemini-3.5-flash", label="doer")

    with patch("dialectica.agent_runtime.run_agent", fake):
        asyncio.run(Workflow(script).run())
    return captured


@then("the underlying agent's model is the resolved model name")
def model_resolved(model_result):
    assert model_result["agent"].model == "gemini-3.5-flash"


# --- Scenario 13: concurrency cap gates each agent call --------------------


@given("a mocked LLM that records calls in flight", target_fixture="inflight_llm")
def inflight_llm():
    state = {"inflight": 0, "max": 0}

    async def fake(agent, instruction: str) -> str:
        state["inflight"] += 1
        state["max"] = max(state["max"], state["inflight"])
        await asyncio.sleep(0.01)
        state["inflight"] -= 1
        return "ok"

    return fake, state


@when(
    "three agent calls run concurrently under a cap of one",
    target_fixture="inflight_state",
)
def run_capped_agents(inflight_llm):
    fake, state = inflight_llm

    async def script():
        await asyncio.gather(wf.agent("a"), wf.agent("b"), wf.agent("c"))

    with patch("dialectica.agent_runtime.run_agent", fake):
        asyncio.run(Workflow(script, concurrency=1).run())
    return state


@then("no more than one LLM call was ever in flight")
def capped_in_flight(inflight_state):
    assert inflight_state["max"] == 1, f"max in flight was {inflight_state['max']}"


# --- Scenario 14: a waiting pipeline item holds no concurrency slot --------


@given(
    "a concurrency cap of one and a pipeline item that waits for its sibling",
    target_fixture="waiting_llm",
)
def waiting_llm():
    async def fake(agent, instruction: str) -> str:
        return instruction

    return fake


@when("the pipeline runs both items", target_fixture="waiting_result")
def run_waiting_pipeline(waiting_llm):
    async def script():
        sibling_done = asyncio.Event()

        async def stage(prev, item, index):
            if index == 0:
                # Item 0 idles until item 1 finishes; under a chain-held slot
                # with cap 1 this deadlocks — the cap must gate agent() only.
                await sibling_done.wait()
                return await wf.agent("first-after-sibling")
            result = await wf.agent("second")
            sibling_done.set()
            return result

        return await wf.pipeline(["a", "b"], stage)

    with patch("dialectica.agent_runtime.run_agent", waiting_llm):
        return asyncio.run(
            asyncio.wait_for(Workflow(script, concurrency=1).run(), timeout=2)
        )


@then("both items complete because waiting held no slot")
def waiting_completed(waiting_result):
    assert waiting_result == ["first-after-sibling", "second"]


# --- Scenario 15/16: budget meters API-reported token usage ----------------


@given(
    "a mocked LLM that reports token usage on each response",
    target_fixture="usage_llm",
)
def usage_llm():
    async def fake(agent, instruction: str) -> str:
        return AgentResponse(
            "ok", TokenUsage(prompt_tokens=100, output_tokens=40, total_tokens=140)
        )

    return fake


@when("two agent calls run in a workflow", target_fixture="usage_budget")
def run_two_usage_calls(usage_llm):
    async def script():
        await wf.agent("a")
        await wf.agent("b")
        return wf.budget()

    with patch("dialectica.agent_runtime.run_agent", usage_llm):
        return asyncio.run(Workflow(script).run())


@then("the budget records the summed prompt, output, and total tokens")
def usage_summed(usage_budget):
    assert usage_budget.usage() == TokenUsage(
        prompt_tokens=200, output_tokens=80, total_tokens=280
    )
    assert usage_budget.spent_tokens() == 80
    assert usage_budget.spent_calls() == 2


@when(
    "a second agent call starts after the first spends the token budget",
    target_fixture="token_budget_result",
)
def run_token_budget(usage_llm):
    state = {"raised": None}

    async def script():
        await wf.agent("first")  # spends the full 40-output-token budget
        try:
            await wf.agent("second")
        except BudgetExhausted as e:
            state["raised"] = str(e)
        return state

    with patch("dialectica.agent_runtime.run_agent", usage_llm):
        return asyncio.run(
            Workflow(script, budget_total=40, budget_unit="tokens").run()
        )


@then("the token budget raises BudgetExhausted")
def token_budget_raised(token_budget_result):
    assert token_budget_result["raised"] is not None
    assert "40/40" in token_budget_result["raised"]
    assert "output tokens" in token_budget_result["raised"]


# --- Scenario 17: plain-string responses keep the token meter at zero ------


@when("two agent calls run with plain-string responses", target_fixture="plain_budget")
def run_two_plain_calls(prose_llm):
    async def script():
        await wf.agent("a")
        await wf.agent("b")
        return wf.budget()

    with patch("dialectica.agent_runtime.run_agent", prose_llm):
        return asyncio.run(Workflow(script).run())


@then("the budget records zero tokens spent")
def plain_zero_tokens(plain_budget):
    assert plain_budget.spent_tokens() == 0
    assert plain_budget.usage() == TokenUsage()
    assert plain_budget.spent_calls() == 2


# --- Scenario 18: schema re-asks meter every underlying call ---------------


@given(
    "a mocked LLM that returns unparseable JSON with token usage",
    target_fixture="bad_usage_llm",
)
def bad_usage_llm():
    async def fake(agent, instruction: str) -> str:
        return AgentResponse(
            "not json at all",
            TokenUsage(prompt_tokens=10, output_tokens=5, total_tokens=15),
        )

    return fake


@when(
    "agent retries a schema on usage-reporting responses",
    target_fixture="reask_budget",
)
def run_agent_reask_usage(bad_usage_llm):
    async def script():
        await wf.agent("judge", schema=_Item, label="judge", max_attempts=2)
        return wf.budget()

    with patch("dialectica.agent_runtime.run_agent", bad_usage_llm):
        return asyncio.run(Workflow(script).run())


@then("the budget records the token usage of every re-ask")
def reask_usage_metered(reask_budget):
    # max_attempts=2 -> the initial call plus one re-ask, both metered; the
    # calls meter counts agent() entries, not underlying LLM calls.
    assert reask_budget.spent_tokens() == 10
    assert reask_budget.usage() == TokenUsage(
        prompt_tokens=20, output_tokens=10, total_tokens=30
    )
    assert reask_budget.spent_calls() == 1


# --- plain regression guards for the budget unit ---------------------------


def test_budget_rejects_unknown_unit():
    with pytest.raises(ValueError, match="unit"):
        Budget(total=10, unit="dollars")


# --- Scenario 19: workflow() standalone ------------------------------------


@given("a mocked LLM", target_fixture="simple_llm")
def simple_llm():
    state = {"calls": 0}

    async def fake(agent, instruction: str) -> str:
        state["calls"] += 1
        return "child-result"

    return fake, state


@when("workflow runs a script standalone", target_fixture="standalone_result")
def run_workflow_standalone(simple_llm):
    fake, state = simple_llm

    async def child():
        return await wf.agent("child-task")

    with patch("dialectica.agent_runtime.run_agent", fake):
        result = asyncio.run(wf.workflow(child))
    return result, state


@then("the script result is returned")
def standalone_result_returned(standalone_result):
    assert standalone_result[0] == "child-result"


@then("the run used one agent call")
def standalone_one_call(standalone_result):
    assert standalone_result[1]["calls"] == 1


# --- Scenario 20: workflow() joins outer budget ----------------------------


@when(
    "a child workflow runs inside an outer workflow with a budget of two calls",
    target_fixture="nested_budget_result",
)
def run_nested_workflow(simple_llm):
    fake, _ = simple_llm

    async def child():
        await wf.agent("child-one")
        return await wf.agent("child-two")

    async def outer():
        result = await wf.workflow(child)
        return result, wf.budget().spent()

    with patch("dialectica.agent_runtime.run_agent", fake):
        return asyncio.run(Workflow(outer, budget_total=2).run())


@then("the outer budget records 2 calls spent")
def nested_budget_charged(nested_budget_result):
    _, spent = nested_budget_result
    assert spent == 2


# --- Scenario 21: double workflow() nesting raises -------------------------


@when("workflow is called inside a child workflow", target_fixture="nesting_error")
def run_double_nested_workflow(simple_llm):
    fake, _ = simple_llm

    async def grandchild():
        return await wf.workflow(lambda: wf.agent("too deep"))

    async def child():
        return await wf.workflow(grandchild)

    async def outer():
        return await wf.workflow(child)

    with patch("dialectica.agent_runtime.run_agent", fake):
        try:
            asyncio.run(Workflow(outer).run())
            return None
        except RuntimeError as e:
            return e


@then("it raises a nesting limit error")
def nesting_limit_raised(nesting_error):
    assert nesting_error is not None
    assert "one level" in str(nesting_error).lower()


# --- Scenario 22: workflow() passes args -----------------------------------


@when(
    "workflow runs with args inside an outer workflow",
    target_fixture="workflow_args_result",
)
def run_workflow_with_args(simple_llm):
    fake, _ = simple_llm
    holder = {"seen": None}

    async def child():
        holder["seen"] = wf.args()
        return "ok"

    async def outer():
        await wf.workflow(child, args={"key": "value"})
        return holder["seen"]

    with patch("dialectica.agent_runtime.run_agent", fake):
        return asyncio.run(Workflow(outer).run())


@then("the child script reads the passed args")
def workflow_args_visible(workflow_args_result):
    assert workflow_args_result == {"key": "value"}


# --- Scenario 23: parallel item cap ----------------------------------------


@when("parallel runs 4097 thunks", target_fixture="item_cap_error")
def run_parallel_cap():
    async def script():
        return await wf.parallel([lambda: None for _ in range(4097)])

    try:
        asyncio.run(Workflow(script).run())
        return None
    except ValueError as e:
        return e


@then("it raises an item cap error")
def item_cap_raised(item_cap_error):
    assert item_cap_error is not None
    assert "4096" in str(item_cap_error)


# --- Scenario 24: lifetime agent cap ---------------------------------------


@when("agent is called 1001 times in one run", target_fixture="agent_cap_error")
def run_agent_cap():
    async def fake(agent, instruction: str) -> str:
        return "ok"

    async def script():
        for i in range(1001):
            await wf.agent(f"call-{i}")

    with patch("dialectica.agent_runtime.run_agent", fake):
        try:
            asyncio.run(Workflow(script).run())
            return None
        except wf.WorkflowAgentCapExceeded as e:
            return e


@then("it raises WorkflowAgentCapExceeded")
def agent_cap_raised(agent_cap_error):
    assert agent_cap_error is not None
    assert "1000" in str(agent_cap_error)


# --- Scenario 25: registered workflow name --------------------------------


@given('a mocked LLM and a registered workflow "demo"', target_fixture="registered_llm")
def registered_llm():
    async def fake(agent, instruction: str) -> str:
        return "ok"

    wf.register_workflow("demo", _demo_script)
    return fake


async def _demo_script():
    return await wf.agent("registered-task")


@when("workflow runs the registered name", target_fixture="registered_result")
def run_registered(registered_llm):
    with patch("dialectica.agent_runtime.run_agent", registered_llm):
        return asyncio.run(wf.workflow("demo"))


@then("the registered script result is returned")
def registered_ok(registered_result):
    assert registered_result == "ok"


# --- Scenario 26: meta phase mismatch --------------------------------------


@given("a mocked LLM and mismatched meta phases", target_fixture="meta_llm")
def meta_llm():
    async def fake(agent, instruction: str) -> str:
        return "ok"

    return fake


@when("the meta workflow runs", target_fixture="meta_error")
def run_meta_mismatch(meta_llm):
    meta = {
        "name": "demo",
        "description": "demo workflow",
        "phases": [{"title": "Wrong"}],
    }

    async def script():
        wf.phase("Gather")
        await wf.agent("x")

    with patch("dialectica.agent_runtime.run_agent", meta_llm):
        try:
            asyncio.run(Workflow(script, meta=meta).run())
            return None
        except wf.WorkflowMetaError as e:
            return e


@then("it raises WorkflowMetaError")
def meta_mismatch(meta_error):
    assert meta_error is not None


# --- Scenario 27: worktree isolation ---------------------------------------


@given("a mocked LLM and a git repository", target_fixture="git_repo")
def git_repo(tmp_path, monkeypatch):
    async def fake(agent, instruction: str) -> str:
        return "ok"

    monkeypatch.chdir(tmp_path)
    subprocess = __import__("subprocess")
    subprocess.run(["git", "init"], check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], check=True, capture_output=True
    )
    (tmp_path / "README").write_text("hi\n")
    subprocess.run(["git", "add", "README"], check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], check=True, capture_output=True)
    return fake, tmp_path


@when(
    "agent runs with worktree isolation and no file changes",
    target_fixture="worktree_paths",
)
def run_worktree_agent(git_repo):
    fake, root = git_repo
    worktrees_before = (
        set((root / ".git/worktrees").iterdir())
        if (root / ".git/worktrees").exists()
        else set()
    )

    async def script():
        await wf.agent("inspect", isolation="worktree", label="inspect")

    with patch("dialectica.agent_runtime.run_agent", fake):
        asyncio.run(Workflow(script, journal_dir=root / "journals").run())

    worktrees_after = (
        set((root / ".git/worktrees").iterdir())
        if (root / ".git/worktrees").exists()
        else set()
    )
    return worktrees_before, worktrees_after


@then("the worktree directory is removed")
def worktree_removed(worktree_paths):
    before, after = worktree_paths
    assert len(after) <= len(before)


# --- Scenario 28-31: per-step access lists (Fugu-Ultra-style isolation) ----


@given(
    "a mocked LLM that echoes the instruction it received",
    target_fixture="echo_instructions",
)
def echo_instructions():
    seen: list[str] = []

    async def fake(agent, instruction: str) -> str:
        seen.append(instruction)
        return instruction

    return fake, seen


def _run_two_agents_with_access(fake, sees):
    async def script():
        await wf.agent("first step secret: ROSEBUD", label="first")
        return await wf.agent("second step task", label="second", sees=sees)

    with patch("dialectica.agent_runtime.run_agent", fake):
        return asyncio.run(Workflow(script).run())


@when(
    "a second agent runs after a first without an access list",
    target_fixture="echo_seen",
)
def second_agent_isolated(echo_instructions):
    fake, seen = echo_instructions
    _run_two_agents_with_access(fake, sees=None)
    return seen


@then("the second agent's instruction does not contain the first agent's output")
def second_isolated(echo_seen):
    assert "ROSEBUD" in echo_seen[0]
    assert "ROSEBUD" not in echo_seen[1]


@when(
    "a second agent runs after a first with the first's label in its access list",
    target_fixture="echo_seen",
)
def second_agent_sees_first(echo_instructions):
    fake, seen = echo_instructions
    _run_two_agents_with_access(fake, sees=["first"])
    return seen


@then("the second agent's instruction contains the first agent's output")
def second_sees_first(echo_seen):
    assert "ROSEBUD" in echo_seen[1]


@when(
    "a third agent runs seeing the first but not the second step",
    target_fixture="echo_seen",
)
def third_agent_sees_first_only(echo_instructions):
    fake, seen = echo_instructions

    async def script():
        await wf.agent("first step secret: ROSEBUD", label="first")
        await wf.agent("second step secret: LILAC", label="second")
        return await wf.agent("third step task", label="third", sees=["first"])

    with patch("dialectica.agent_runtime.run_agent", fake):
        asyncio.run(Workflow(script).run())
    return seen


@then("the third agent's instruction contains the first agent's output")
def third_sees_first(echo_seen):
    assert "ROSEBUD" in echo_seen[-1]


@then("the third agent's instruction does not contain the second agent's output")
def third_isolated_from_second(echo_seen):
    assert "LILAC" not in echo_seen[-1]


@when(
    "an agent runs with an access list naming a step that never ran",
    target_fixture="unknown_access_result",
)
def agent_unknown_access_label(echo_instructions):
    fake, _ = echo_instructions

    async def script():
        return await wf.agent("do it", label="solo", sees=["ghost"])

    with patch("dialectica.agent_runtime.run_agent", fake):
        return asyncio.run(Workflow(script).run())


@then("it runs and returns its own prompt without error")
def unknown_access_ok(unknown_access_result):
    assert unknown_access_result == "do it"


# --- Scenario 32: regression — JSON instruction not suppressed by sees= context ---


class _Decision(BaseModel):
    choice: str


@when(
    "a schema agent sees a prior step whose output contains the word json",
    target_fixture="schema_sees_seen",
)
def schema_agent_sees_json_step(echo_instructions):
    fake, seen = echo_instructions

    async def script():
        await wf.agent("Explain json parsing libraries", label="gather")
        return await wf.agent(
            "Decide", schema=_Decision, label="decide", sees=["gather"]
        )

    with patch("dialectica.agent_runtime.run_agent", fake):
        asyncio.run(Workflow(script).run())
    return seen


@then("the schema agent's instruction contains the JSON-format directive")
def schema_agent_has_json_directive(schema_sees_seen):
    assert "Return your answer as a single JSON object." in schema_sees_seen[-1]
