"""Reference pattern: an agentic (tool-using) workflow stage.

DEMOTED FROM THE SHIPPED API (was ``dialectica.create_agentic_engine``). The
capability this pattern demonstrates — injecting ``tools=`` into
``wf.agent()`` so a stage acts instead of only rearranging text — is now a
first-class ``workflow.py`` primitive; there is nothing left that requires a
dedicated engine class. This script is kept as a runnable reference for the
pattern, not as shipped library code (see README "Patterns").

Measured result (``evals/agentic_eval.py``, small model, hidden-oracle
benchmark): 8/8 vs a single call's 0/8. The win requires genuinely injecting
tools — a workflow stage with no tools is just a pure-LLM call and ties/loses
like every other scaffold in this repo.
"""

from typing import Any

from dialectica import workflow as wf

AGENTIC_SYSTEM_PROMPT = """You are a capable agent that completes a task by USING TOOLS, not by guessing.

How to work:
- Plan briefly, then act: call a tool, read its result, and decide the next step from what you actually observe.
- Never assume a file's contents, a command's output, or that the task is done — verify with the tools.
- Iterate: if a check fails, use the failure to fix the cause, then re-check. Repeat until the task genuinely passes.
- Stop when the task's success condition is objectively met (or you are certain it cannot be), then state the outcome plainly.

{instructions}"""


class AgenticEngine:
    """Runs a tool-using workflow stage on a task; the injected tools do the acting."""

    def __init__(
        self,
        task: str,
        tools: list,
        model_config: str | None,
        instructions: str,
    ):
        self.task = task
        self.tools = tools
        self.model_config = model_config
        self.instructions = instructions

    async def run(self) -> dict[str, Any]:
        """Execute the stage (ADK drives the tool-call loop) and return the result.

        Side effects (file edits, etc.) happen through the tools, so the
        caller checks the objective outcome (e.g. run the tests) after this
        returns.
        """

        async def script() -> str:
            return await wf.agent(
                self.task,
                tools=self.tools,
                model=self.model_config,
                instructions=AGENTIC_SYSTEM_PROMPT.format(
                    instructions=self.instructions
                ).strip(),
                label="Agent",
            )

        answer = await wf.workflow(script)
        return {"final_answer": answer}


def create_agentic_engine(
    task: str,
    tools: list,
    model_config: str | None = None,
    instructions: str = "",
) -> AgenticEngine:
    """Wire a tool-using workflow stage for ``task``.

    ``tools`` is a list of plain callables (or ADK ``FunctionTool``s) the
    agent may call; ADK auto-derives their schemas. ``instructions`` appends
    task-specific guidance (e.g. how to run the tests, what "done" means).
    """
    return AgenticEngine(task, tools, model_config, instructions)
