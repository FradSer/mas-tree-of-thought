"""Reference pattern: Tree-of-Thoughts + GAN-style adversarial refinement.

DEMOTED FROM THE SHIPPED API (was ``dialectica.create_engine``/
``create_coordinator``). Measured "dominated" at matched compute — never
wins a single matchup against a single call, best-of-N, or flat self-refine
(README findings #2 / #3); on Game-of-24 (ToT's own canonical
benchmark) a faithful ToT scored 14/15 and LOST to a single call's 15/15 at
~34x the cost. Kept as a runnable reference for the beam-search+GAN pattern,
not as a shipped engine.

Written in ``workflow.py``'s own compositional idiom — plain functions/
closures over ``agent()``/``parallel()``, explicit local state (a node dict +
an active-beam list) — rather than the original's Protocol-based plugin
system (``Generator``/``Evaluator``/``Selector``/``Synthesizer``), which is
not ported: this pattern hardcodes one shape, consistent with "kept for
study, not extension." The cross-call parse-failure circuit breaker is also
not ported (acceptable simplification for study-only code) —
``wf.agent(schema=...)``'s own per-call retry still applies. ``structured_output``
is accepted for signature parity but always uses schema-enforced scoring; the
gemma-4-26b-a4b enforced-JSON quirk workaround from the original engine is
not reproduced here.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from dialectica import workflow as wf

from ._scoring import DEFAULT_CRITERIA, Verdict, build_scoring_prompt, clamp_score

logger = logging.getLogger(__name__)

# --- Generation --------------------------------------------------------

# Matches a numbered (1. / 1)) or bulleted (-, *, •) list-item start.
_ITEM_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+(.*)$")


def _parse_list(response: str) -> list[str]:
    """Parse a numbered/bulleted list from an agent response.

    Continuation lines are accumulated into the current item, so multi-line
    entries keep their full body. Falls back to one-item-per-line.
    """
    items: list[list[str]] = []
    for raw in response.strip().splitlines():
        if not raw.strip():
            continue
        match = _ITEM_MARKER.match(raw)
        if match:
            items.append([match.group(1).strip()])
        elif items:
            items[-1].append(raw.strip())

    parsed = [" ".join(parts).strip() for parts in items]
    parsed = [p for p in parsed if p]
    if parsed:
        return parsed
    return [line.strip() for line in response.strip().splitlines() if line.strip()]


STRATEGY_PROMPT = """Generate 3-5 distinct initial strategies to solve this problem:

**Problem:**
{problem}

**Requirements:**
- Each strategy should represent a fundamentally different approach
- Be specific and actionable
- Consider different perspectives and trade-offs

**Output Format:**
Return a FLAT numbered list — one strategy per line, starting with "1. ", "2. ", etc.
Each strategy is a single self-contained line. Do NOT use sub-bullets, nested
lists, headings, code blocks, or multi-paragraph explanations.
"""

CHILD_PROMPT = """Generate 2-4 specific next steps or refinements for this thought:

**Parent Thought:**
{parent}

**Context:**
- Problem: {problem}
- Depth: {depth}

**Requirements:**
- Each child should be a concrete step forward
- Build on the parent thought, don't just restate it
- Consider different angles or sub-problems

**Output Format:**
Return a FLAT numbered list — one child thought per line, starting with "1. ", "2. ", etc.
Each item is a single self-contained line. Do NOT use sub-bullets, nested lists,
headings, code blocks, or multi-paragraph explanations.
"""

REFINE_PROMPT = """Refine the following thought based on critical feedback:

**Original Thought:**
{thought}

**Context:**
{context}

**Identified Flaws:**
{flaws}

**Suggestions for Improvement:**
{suggestions}

**Discriminator's Reasoning:**
{reasoning}

**Your Task:**
Address the identified flaws and incorporate the suggestions to create an improved version of the thought. Maintain the core intent while strengthening weak points.

**Output Format:**
Provide the refined thought directly, without introductory text or commentary.
"""

SYNTHESIS_PROMPT = """Synthesize a comprehensive solution from the following high-quality thoughts:

**Original Problem:**
{problem}

**Top Thoughts:**
{thoughts}

**Your Task:**
1. Identify common themes and complementary insights
2. Resolve any conflicts between different approaches
3. Create a coherent, actionable solution
4. Structure the answer clearly with sections if appropriate

**Output:**
Provide the synthesized solution directly, without additional commentary.
"""

_NO_THOUGHTS = "Unable to generate sufficient high-quality thoughts for synthesis."


@dataclass
class Node:
    """One node in the thought tree — a plain dataclass, not a validated schema."""

    thought_id: str
    parent_id: str | None
    thought: str
    depth: int
    score: float | None = None
    rounds: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


def _score_of(node: Node) -> float:
    return node.score if node.score is not None else 0.0


class Coordinator:
    """Beam search over a thought tree, with GAN-style refinement per node.

    Phases: Initialize (root -> strategies -> score) -> Explore (beam search
    loop) -> Synthesize. Sibling expansions/evaluations run concurrently via
    ``wf.parallel``.
    """

    def __init__(
        self,
        problem: str,
        max_depth: int = 2,
        beam_width: int = 2,
        max_gan_rounds: int = 2,
        score_threshold: float = 7.0,
        synthesizer_model: str | None = None,
        gan_score_threshold: float | None = None,
        criteria: str | None = None,
        structured_output: bool = True,
    ):
        self.problem = problem
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.max_gan_rounds = max_gan_rounds
        self.score_threshold = score_threshold
        self.synthesizer_model = synthesizer_model
        self.gan_score_threshold = (
            gan_score_threshold if gan_score_threshold is not None else score_threshold
        )
        self.criteria = criteria if criteria is not None else DEFAULT_CRITERIA
        self.structured_output = structured_output
        self.thought_tree: dict[str, Node] = {}
        self.active_beam: list[str] = []

    async def _score_and_refine(self, node: Node, parent_thought: str) -> None:
        """GAN-style loop: critique -> refine -> re-score, in place on ``node``.

        Keeps the best-scoring round, not the last (refinement is not
        monotonic), mirroring the original ``AdversarialEvaluator``.
        """
        context = {
            "problem": self.problem,
            "parent_thought": parent_thought,
            "depth": node.depth,
        }
        current = node.thought
        best_score: float | None = None
        best_thought = current

        for round_num in range(1, self.max_gan_rounds + 1):
            verdict: Verdict | None = await wf.agent(
                build_scoring_prompt(current, context, self.criteria),
                schema=Verdict,
                label="discriminator",
            )
            score = clamp_score(verdict)
            node.history.append(
                {"round": round_num, "thought": current, "score": score}
            )
            if best_score is None or score > best_score:
                best_score, best_thought = score, current

            if verdict is not None and verdict.should_terminate:
                break
            if score >= self.gan_score_threshold:
                break
            if round_num < self.max_gan_rounds and verdict is not None:
                current = (
                    await wf.agent(
                        REFINE_PROMPT.format(
                            thought=current,
                            context="\n".join(
                                f"- {k}: {v}" for k, v in context.items() if v
                            ),
                            flaws="\n".join(f"- {f}" for f in verdict.flaws),
                            suggestions="\n".join(
                                f"- {s}" for s in verdict.suggestions
                            ),
                            reasoning=verdict.reasoning,
                        ),
                        label="generator",
                    )
                ).strip()

        node.thought = best_thought
        node.score = best_score if best_score is not None else 0.0
        node.rounds = len(node.history)

    async def _expand(self, parent: Node) -> list[str]:
        if parent.depth == 0:
            instruction = STRATEGY_PROMPT.format(problem=self.problem)
        else:
            instruction = CHILD_PROMPT.format(
                problem=self.problem, parent=parent.thought, depth=parent.depth
            )
        response = await wf.agent(instruction, label="generator")
        return _parse_list(response)[:8]

    async def run(self) -> dict[str, Any]:
        """Execute the full search and return the answer plus tree and stats."""

        async def script() -> dict[str, Any]:
            start_time = datetime.now(tz=UTC)

            logger.info("Phase 1: Initializing thought tree")
            await self._initialize()

            logger.info("Phase 2: Exploring with beam search")
            await self._explore()

            logger.info("Phase 3: Synthesizing final answer")
            final_answer = await self._synthesize()

            duration = (datetime.now(tz=UTC) - start_time).total_seconds()
            return {
                "final_answer": final_answer,
                "thought_tree": {
                    k: {
                        "thoughtId": v.thought_id,
                        "parentId": v.parent_id,
                        "thought": v.thought,
                        "depth": v.depth,
                        "evaluationScore": v.score,
                        "adversarialRounds": v.rounds,
                    }
                    for k, v in self.thought_tree.items()
                },
                "best_path": self._best_path(),
                "stats": {
                    "total_thoughts": len(self.thought_tree),
                    "max_depth_reached": max(
                        (n.depth for n in self.thought_tree.values()), default=0
                    ),
                    "duration_seconds": duration,
                },
            }

        return await wf.workflow(script)

    async def _initialize(self) -> None:
        """Phase 1: create the root, expand into strategies, score each."""
        root = Node(thought_id="root", parent_id=None, thought=self.problem, depth=0)
        self.thought_tree["root"] = root

        strategies: list[str] = []
        for attempt in range(1, 4):
            strategies = await self._expand(root)
            if strategies:
                break
            logger.warning(
                "Initial strategy generation returned nothing (attempt %d/3)", attempt
            )

        strategy_ids = []
        for i, content in enumerate(strategies):
            node = Node(
                thought_id=f"root_s{i}", parent_id="root", thought=content, depth=1
            )
            self.thought_tree[node.thought_id] = node
            strategy_ids.append(node.thought_id)

        # Siblings are independent, so they are scored concurrently.
        await wf.parallel(
            [
                (
                    lambda sid=sid: self._score_and_refine(
                        self.thought_tree[sid], self.problem
                    )
                )
                for sid in strategy_ids
            ]
        )
        self.active_beam = [
            sid
            for sid in strategy_ids
            if _score_of(self.thought_tree[sid]) >= self.score_threshold
        ]

        # Don't stall the whole run if nothing cleared the bar: seed
        # exploration with the top-scoring strategies instead.
        if not self.active_beam and strategy_ids:
            ranked = sorted(
                strategy_ids,
                key=lambda sid: _score_of(self.thought_tree[sid]),
                reverse=True,
            )
            self.active_beam = ranked[: self.beam_width]

    async def _explore(self) -> None:
        """Phase 2: beam search — select, expand, score, repeat."""
        for iteration in range(1, self.max_depth):
            if not self.active_beam:
                break

            frontier_ids = sorted(
                self.active_beam,
                key=lambda nid: _score_of(self.thought_tree[nid]),
                reverse=True,
            )[: self.beam_width]
            parents = [
                self.thought_tree[nid]
                for nid in frontier_ids
                if self.thought_tree[nid].depth < self.max_depth
            ]

            expansions = await wf.parallel(
                [(lambda p=p: self._expand(p)) for p in parents]
            )

            children: list[tuple[str, str]] = []  # (child_id, parent_thought)
            for parent, contents in zip(parents, expansions):
                for i, content in enumerate(contents or []):
                    child_id = f"{parent.thought_id}_c{i}"
                    self.thought_tree[child_id] = Node(
                        thought_id=child_id,
                        parent_id=parent.thought_id,
                        thought=content,
                        depth=parent.depth + 1,
                    )
                    children.append((child_id, parent.thought))

            await wf.parallel(
                [
                    (
                        lambda cid=cid, pt=pt: self._score_and_refine(
                            self.thought_tree[cid], pt
                        )
                    )
                    for cid, pt in children
                ]
            )
            self.active_beam = [
                cid
                for cid, _ in children
                if _score_of(self.thought_tree[cid]) >= self.score_threshold
            ]
            if not self.active_beam:
                logger.info("No candidates meet threshold, stopping exploration")
                break

    async def _synthesize(self) -> str:
        scored = [n for n in self.thought_tree.values() if n.score is not None]
        scored.sort(key=_score_of, reverse=True)
        top = scored[:10]
        if not top:
            return _NO_THOUGHTS

        thoughts_text = "\n\n".join(
            f"**Thought (Score: {n.score}/10):**\n{n.thought}" for n in top
        )
        response = await wf.agent(
            SYNTHESIS_PROMPT.format(problem=self.problem, thoughts=thoughts_text),
            model=self.synthesizer_model,
            label="synthesizer",
        )
        return response.strip()

    def _best_path(self) -> list[str]:
        """Get the path from root to the highest-scoring evaluated node."""
        scored = [n for n in self.thought_tree.values() if n.score is not None]
        if not scored:
            return ["root"] if "root" in self.thought_tree else []

        best = max(scored, key=_score_of)
        path = []
        current_id: str | None = best.thought_id
        while current_id:
            path.append(current_id)
            current = self.thought_tree.get(current_id)
            current_id = current.parent_id if current else None
        return list(reversed(path))


def create_coordinator(
    problem: str,
    max_depth: int = 2,
    beam_width: int = 2,
    max_gan_rounds: int = 2,
    score_threshold: float = 7.0,
    synthesizer_model: str | None = None,
    gan_score_threshold: float | None = None,
    criteria: str | None = None,
    structured_output: bool = True,
) -> Coordinator:
    """Wire a Coordinator with the default beam-search + GAN-refinement pattern.

    See module docstring — measured dominated, kept for study and back-compat
    with historical eval numbers, not recommended for quality.
    """
    logger.info("Creating coordinator for problem: %s...", problem[:50])
    return Coordinator(
        problem=problem,
        max_depth=max_depth,
        beam_width=beam_width,
        max_gan_rounds=max_gan_rounds,
        score_threshold=score_threshold,
        synthesizer_model=synthesizer_model,
        gan_score_threshold=gan_score_threshold,
        criteria=criteria,
        structured_output=structured_output,
    )


# Canonical Dialectica names. The Coordinator/create_coordinator names are kept
# as aliases for backward compatibility with the demoted engine's call sites.
create_engine = create_coordinator
Engine = Coordinator


__all__ = [
    "Coordinator",
    "Engine",
    "create_coordinator",
    "create_engine",
]
