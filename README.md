# Multi-Tool Agent with Sequential Thinking

This project implements a multi-agent system coordinated by a root agent (`sequential_thinking_coordinator`). The coordinator manages a sequence of "thoughts" to solve a complex problem.

## Core Concepts

*   **Sequential Thinking:** Problems are broken down into a series of thoughts, managed by the `sequentialthinking` tool. This allows for structured reasoning, revisions, and branching logic.
*   **Coordinator Agent:** The `sequential_thinking_coordinator` plans each thought, logs its metadata using the `sequentialthinking` tool, delegates the core task of the thought to a specialist agent, integrates the result, and decides the next step.
*   **Specialist Agents:** A team of agents (Planner, Researcher, Analyzer, Critic, Synthesizer) each handle specific types of sub-tasks delegated by the coordinator. Each specialist focuses only on its assigned task for a given thought.
*   **Agent Tools:** The coordinator uses `AgentTool` instances to interact with the specialist agents.

## Workflow

1.  The user provides an initial problem.
2.  The Coordinator plans the first thought and logs it via `sequentialthinking`.
3.  Based on the thought's content, the Coordinator delegates the task to the most appropriate specialist agent (e.g., Planner, Analyzer).
4.  The specialist executes its task and returns the result.
5.  The Coordinator integrates the result and plans the next thought (which could be the next step, a revision, or a branch).
6.  Steps 2-5 repeat until the problem is solved (`nextThoughtNeeded` is false).
7.  The Coordinator synthesizes the entire thought process into a final answer.

## Setup and Usage

(Instructions on how to set up and run the agent would go here)
