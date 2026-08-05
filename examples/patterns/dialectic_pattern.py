"""Reference pattern: thesis → antithesis → synthesis spiral (the "dialectic").

DEMOTED FROM THE SHIPPED API (was ``dialectica.create_dialectic_engine``).
Measured result: ties/loses a prompt-matched single call (0-3-2) — a
pure-LLM scaffold, auditable trace only, not a quality win (see README
"Evaluation"). Kept as a runnable reference for how a self-contained
multi-round LLM scaffold looks when expressed as a ``workflow.py`` script
instead of a dedicated engine class — this module is already close to what
``workflow.py``'s own docstring calls "a fixed workflow script hardcoded into
a class", so the port needed no structural changes, only swapping
``agent_runtime.run_agent(agent, ...)`` calls for ``wf.agent(model=..., ...)``.
"""

import logging
from typing import Any, Optional

from dialectica import workflow as wf

from ._scoring import DEFAULT_CRITERIA, Verdict, build_scoring_prompt, clamp_score

logger = logging.getLogger(__name__)

# The proposer plays three single-answer roles in the dialectic (name the
# tension, give one committed thesis, give one committed rival). The base
# Generator persona tells it to spray "diverse thought branches" — the wrong
# instinct here. This reframes it: commit to one whole position per task;
# diversity comes from opposition between rounds, not variation within an answer.
DIALECTIC_PROPOSER_CONTEXT = """You are operating inside a dialectic, not a breadth-first search. Each task asks for exactly one thing: a single tension judgment, one committed solution, or one committed counter-solution. Give precisely that — take a clear, whole position and own it; do not hedge to a noncommittal middle or pad the answer with loosely-related alternatives. Diversity here comes from genuine opposition between rounds, not from listing variations inside one answer."""

# Full rival solutions are multi-paragraph and often carry their own internal
# numbered steps, so they cannot be recovered as a flat numbered list (a list
# parser shatters a rival on its own "1./2." markers). An explicit delimiter
# between complete alternatives keeps each one whole.
RIVAL_DELIMITER = "===NEXT==="

TENSION_PROMPT = """Before solving, judge whether this problem has a genuine dialectical tension — a real trade-off or opposition where a one-sided answer goes wrong.

**Problem:**
{problem}

If there IS such a tension, reply with ONE line:
X vs Y — <one sentence on why this opposition is the crux>
If the problem has a single correct or determinate answer with no real trade-off (a factual, computational, or well-defined technical task), reply with exactly:
NONE"""

THESIS_PROMPT = """Give your single best solution to this problem — the complete answer an expert would deliver: correct, comprehensive, specific, and actionable, structured with clear sections where it helps.

**Problem:**
{problem}

**What counts as a strong solution (hold yourself to this):**
{criteria}

Commit to a clear position and cover the problem in full — this answer must stand on its own as a strong, complete solution. Provide it directly."""

ANTITHESIS_PROMPT = """A solution (the thesis) has been proposed for a problem whose core tension is:

**Core tension:** {tension}

The thesis takes one side of this tension. Do NOT critique the thesis or list its flaws. Become its dialectical opposite: propose {n} COMPLETE alternative solution(s) to the same problem. The first must fully commit to the OTHER side of the core tension above; any further ones oppose along different axes (e.g. simplicity vs power, short- vs long-term, centralize vs distribute).

Each must be a full, standalone rival solution that a smart advocate of the opposite value would genuinely champion and stake their reputation on — a real competing solution, not a comment on the thesis.

**Problem:**
{problem}

**Thesis:**
{thesis}

**What counts as a strong solution (each rival must hold up to this too):**
{criteria}
{prior_block}
Give each rival as a complete, self-contained solution. If you propose more than one, separate them with a line containing only {delimiter} — do not use a numbered list across rivals (each rival may have its own internal steps)."""

SYNTHESIS_PROMPT = """You are resolving a dialectic: a thesis and the strongest rival solution(s) to it, around this core tension:

**Core tension:** {tension}

**Problem:**
{problem}

**Thesis (one solution):**
{thesis}

**Antithesis (rival solution{plural} built on opposing principles):**
{antithesis}

Produce a SYNTHESIS that transcends the rivalry: take what is right in the thesis AND in each rival, resolve the tension between their underlying principles, and deliver a solution stronger than any of them alone. Do not pick a winner or staple them together — integrate the conflicting truths into a higher solution that none of them held.

Your synthesis must DOMINATE what a single expert writes on a first pass — that is the bar it is measured against:
- Pick ONE binding decision and commit to it. Do NOT enumerate every option — a single-pass answer already does that. Yours wins by sharpness, not coverage.
- Lead with the single sharpest recommendation and the precise trigger that decides it (a measurable condition, not "when ready").
- Resolve the core tension explicitly: name the condition under which each side wins, and make a decisive recommendation for THIS problem's context.
- Name the non-obvious failure mode a naive one-sided answer misses — and show concretely how this solution structurally avoids it.
- Carry forward the specific numbers, steps, and trade-offs from the thesis and rivals; do NOT abstract them into vagueness.

**What makes a synthesis better (judge yourself against this):**
{criteria}

Provide the synthesized solution directly."""


class DialecticEngine:
    """Runs the understand → thesis → antithesis → synthesis spiral."""

    def __init__(
        self,
        problem: str,
        criteria: str = DEFAULT_CRITERIA,
        max_rounds: int = 3,
        perspectives: int = 1,
        model_config: Optional[str] = None,
        discriminator_model: Optional[str] = None,
    ):
        self.problem = problem
        self.criteria = criteria
        self.max_rounds = max_rounds
        self.perspectives = max(1, perspectives)
        self.model_config = model_config
        self.discriminator_model = discriminator_model

    async def _score(self, solution: str) -> float:
        # The discriminator must see the problem; the criteria grade
        # completeness and "feasibility under stated constraints", and the
        # constraints live in the problem. wf.agent(schema=Verdict) already
        # re-asks on transient unparseable output, same seam every other
        # workflow script uses.
        verdict: Verdict | None = await wf.agent(
            build_scoring_prompt(solution, {"problem": self.problem}, self.criteria),
            schema=Verdict,
            model=self.discriminator_model,
            label="discriminator",
        )
        return clamp_score(verdict)

    async def _identify_tension(self) -> str:
        """Name the problem's core contradiction before opposing — the cognitive
        step that makes the dialectic understand the problem, and decide whether
        a dialectic is even warranted.
        """
        return (
            await wf.agent(
                TENSION_PROMPT.format(problem=self.problem),
                model=self.model_config,
                instructions=DIALECTIC_PROPOSER_CONTEXT,
                label="proposer",
            )
        ).strip()

    async def _oppose(self, thesis: str, prior: list[str], tension: str) -> list[str]:
        """Surface complete rival solution(s): the first along the core tension,
        later rounds along axes not yet pressed (spiral ascent, not circling).
        """
        if prior:
            prior_block = (
                "\nYou already pressed these oppositions in earlier rounds — do "
                "NOT repeat them; find genuinely different, deeper axes:\n"
                + "\n".join(f"- {a[:200]}" for a in prior)
                + "\nIf no genuinely new, non-redundant axis of opposition "
                "remains — the dialectic has run its course and any further "
                "rival would just rephrase the above — reply with exactly:\n"
                "EXHAUSTED\n"
            )
        else:
            prior_block = ""
        response = await wf.agent(
            ANTITHESIS_PROMPT.format(
                problem=self.problem,
                thesis=thesis,
                n=self.perspectives,
                prior_block=prior_block,
                tension=tension,
                criteria=self.criteria,
                delimiter=RIVAL_DELIMITER,
            ),
            model=self.model_config,
            instructions=DIALECTIC_PROPOSER_CONTEXT,
            label="proposer",
        )
        if self.perspectives == 1:
            return [response.strip()]
        # Split on the explicit delimiter, never a list parser: each rival is a
        # whole multi-line solution that may carry its own internal numbering.
        parts = [p.strip() for p in response.split(RIVAL_DELIMITER) if p.strip()]
        return parts[: self.perspectives] or [response.strip()]

    async def _synthesize(
        self, thesis: str, antitheses: list[str], tension: str
    ) -> str:
        if len(antitheses) == 1:
            antithesis_block, plural = antitheses[0], ""
        else:
            antithesis_block = "\n\n".join(
                f"{i}. {a}" for i, a in enumerate(antitheses, 1)
            )
            plural = "s, along different axes"
        return (
            await wf.agent(
                SYNTHESIS_PROMPT.format(
                    problem=self.problem,
                    thesis=thesis,
                    antithesis=antithesis_block,
                    plural=plural,
                    tension=tension,
                    criteria=self.criteria,
                ),
                model=self.model_config,
                label="synthesizer",
            )
        ).strip()

    async def run(self) -> dict[str, Any]:
        """Execute the dialectic and return the converged answer + trace."""

        async def script() -> dict[str, Any]:
            tension = await self._identify_tension()
            logger.info("Core tension: %s", tension)
            thesis = (
                await wf.agent(
                    THESIS_PROMPT.format(problem=self.problem, criteria=self.criteria),
                    model=self.model_config,
                    instructions=DIALECTIC_PROPOSER_CONTEXT,
                    label="proposer",
                )
            ).strip()

            # No genuine tension (factual/determinate): a dialectic would be
            # theater and waste 3-5x the calls. Return the direct solution —
            # knowing when NOT to dialecticize is part of being intelligent.
            if tension.strip().upper().startswith("NONE"):
                logger.info("No dialectical tension; returning the direct solution.")
                return {
                    "final_answer": thesis,
                    "score": None,
                    "rounds": 0,
                    "perspectives": self.perspectives,
                    "dialecticized": False,
                    "history": [
                        {"role": "tension", "round": 0, "text": tension},
                        {"role": "thesis", "round": 0, "text": thesis, "score": None},
                    ],
                }

            thesis_score = await self._score(thesis)
            history: list[dict[str, Any]] = [
                {"role": "tension", "round": 0, "text": tension},
                {"role": "thesis", "round": 0, "text": thesis, "score": thesis_score},
            ]
            logger.info("Thesis scored %.1f", thesis_score)

            prior_antitheses: list[str] = []
            for round_num in range(1, self.max_rounds + 1):
                antitheses = await self._oppose(thesis, prior_antitheses, tension)

                # Semantic convergence: the adversary concedes it has no
                # genuinely new axis to press. This is the primary,
                # score-independent stop signal.
                if len(antitheses) == 1 and antitheses[0].strip().upper().startswith(
                    "EXHAUSTED"
                ):
                    logger.info("Adversary exhausted; converged by exhaustion.")
                    break

                prior_antitheses.extend(antitheses)
                synthesis = await self._synthesize(thesis, antitheses, tension)
                synth_score = await self._score(synthesis)

                for a in antitheses:
                    history.append(
                        {"role": "antithesis", "round": round_num, "text": a}
                    )
                history.append(
                    {
                        "role": "synthesis",
                        "round": round_num,
                        "text": synthesis,
                        "score": synth_score,
                    }
                )
                logger.info(
                    "Round %d: synthesis %.1f vs thesis %.1f (%d perspective(s))",
                    round_num,
                    synth_score,
                    thesis_score,
                    len(antitheses),
                )

                # Spiral ascent: a synthesis that surpasses the thesis becomes
                # the next thesis; otherwise the dialectic has converged.
                if synth_score <= thesis_score:
                    logger.info("Synthesis did not surpass thesis; converged.")
                    break
                thesis, thesis_score = synthesis, synth_score

            return {
                "final_answer": thesis,
                "score": thesis_score,
                "rounds": history[-1]["round"],
                "perspectives": self.perspectives,
                "dialecticized": True,
                "history": history,
            }

        return await wf.workflow(script)


def create_dialectic_engine(
    problem: str,
    criteria: Optional[str] = None,
    max_rounds: int = 3,
    perspectives: int = 1,
    model_config: Optional[str] = None,
    discriminator_model: Optional[str] = None,
) -> DialecticEngine:
    """Wire a DialecticEngine (the demoted dialectic reference pattern).

    ``perspectives`` widens each round to oppose along that many distinct axes
    (richer synthesis, more calls). ``discriminator_model`` may be stronger,
    but the dialectic is far less sensitive to evaluator quality than beam
    search — the score only gates iteration, it does not select candidates.
    """
    return DialecticEngine(
        problem=problem,
        criteria=criteria or DEFAULT_CRITERIA,
        max_rounds=max_rounds,
        perspectives=perspectives,
        model_config=model_config,
        discriminator_model=discriminator_model,
    )
