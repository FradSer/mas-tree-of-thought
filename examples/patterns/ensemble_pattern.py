"""Reference pattern: AB-MCTS-lite adaptive search over a heterogeneous roster.

DEMOTED FROM THE SHIPPED API (was ``dialectica.create_ensemble_engine``) per
the project's own honesty gate (``evals/ensemble_ablation.py``,
``evals/ensemble_meta_ablation.py``): a blind-pick roster (the float scorer
replaced by a constant) matched the real scorer's performance on open-ended
tasks (3-1 vs the scorer arm's 3-1-2), and both saturated 6/6 on verifiable
code — the measured robustness gain is attributable to roster
HETEROGENEITY, not the scorer's ranking signal. A no-scorer multi-model
best-of-N captures the same gain more honestly. Kept as a runnable reference
for the AB-MCTS-lite pattern (Sakana arXiv 2503.04412), not as a shipped
engine.

Simplifications vs the original engine: model arms are no longer pre-built
``LlmAgent`` objects — ``wf.agent(model=...)`` resolves each roster config on
demand, so the FR6 roster-distinctness warnings (which needed pre-resolved
configs to compare) are not ported; a caller who wants that check can compare
``models`` for duplicates before calling ``create_ensemble_engine``.
"""

import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from dialectica import workflow as wf

logger = logging.getLogger(__name__)

Scorer = Callable[[str], float]

SOLVE_PROMPT = """Solve the following problem. Reason it through, then provide your COMPLETE solution.{format_hint}

{problem}"""

WIDEN_PROMPT = """Solve the following problem. A previous approach scored {score:.2f} — produce a GENUINELY DIFFERENT approach, not a minor variation.

Previous (wrong) answer:
{prev_answer}

Problem:
{problem}{format_hint}"""

REFINE_PROMPT = """Refine the following solution. Identify what can be improved and provide a COMPLETE improved solution.{format_hint}

Problem:
{problem}

Current best solution (score {score:.2f}):
{answer}"""


@dataclass
class _ActionArm:
    """Internal action-selection bandit arm (wider vs deeper)."""

    alpha: float = 1.0
    beta: float = 1.0


@dataclass
class ModelArm:
    """One roster member: a model config string + its Thompson Beta priors."""

    config: str | None
    alpha: float = 1.0
    beta: float = 1.0


@dataclass
class Candidate:
    """A scored answer produced by one ``wf.agent`` call."""

    answer: str
    score: float
    model: str | None  # producing arm's config
    action: str  # "wider" | "deeper"
    depth: int
    parent: int | None = field(default=None)  # index into candidates list


class ThompsonPolicy:
    """Default Beta-Bernoulli bandit: Thompson-sample action and arm.

    ``Policy`` is duck-typed here (two methods, no formal ``typing.Protocol``)
    — tests inject a scripted stand-in with the same shape for determinism.
    """

    def __init__(self, seed: int | None = None, rng: random.Random | None = None):
        self._rng = rng if rng is not None else random.Random(seed)

    def choose_action(
        self, wider: _ActionArm, deeper: _ActionArm, have_candidates: bool
    ) -> str:
        if not have_candidates:
            return "wider"
        w = self._rng.betavariate(wider.alpha, wider.beta)
        d = self._rng.betavariate(deeper.alpha, deeper.beta)
        return "wider" if w >= d else "deeper"

    def choose_arm(self, roster: list[ModelArm]) -> ModelArm:
        draws = [self._rng.betavariate(arm.alpha, arm.beta) for arm in roster]
        return roster[draws.index(max(draws))]


class EnsembleSearchEngine:
    """AB-MCTS-lite adaptive search over a heterogeneous roster.

    Each step chooses via the injectable ``policy`` between a "wider" move
    (fresh candidate, optionally cross-model-hinted) and a "deeper" move
    (refine the current best with the arm that produced it). The scorer ranks
    answers; the first answer to reach ``solved_score`` short-circuits the
    budget.
    """

    def __init__(
        self,
        problem: str,
        scorer: Scorer,
        roster: list[ModelArm],
        max_calls: int = 8,
        solved_score: float = 1.0,
        solution_format: str = "",
        policy: Any | None = None,
    ):
        if scorer is None:
            raise TypeError(
                "scorer is required: EnsembleSearchEngine needs a float-valued "
                "scorer to rank candidates. Wrap a boolean verifier with "
                "'lambda a: 1.0 if v(a)[0] else 0.0'."
            )
        self.problem = problem
        self.scorer = scorer
        self.roster = list(roster)
        self.max_calls = max(1, max_calls)
        self.solved_score = solved_score
        self._format_hint = f" {solution_format}" if solution_format else ""
        self.policy = policy if policy is not None else ThompsonPolicy()
        self._wider_arm = _ActionArm()
        self._deeper_arm = _ActionArm()

    def _find_arm(self, config: str | None) -> ModelArm | None:
        for arm in self.roster:
            if arm.config == config:
                return arm
        return None

    async def run(self) -> dict[str, Any]:
        """Execute the AB-MCTS-lite loop; return the best candidate found.

        Return shape mirrors ``create_repair_engine``'s:
        ``{final_answer, passed, attempts, history, best_score}``.
        ``history`` entries: ``{call, action, model, score, failed}``.
        """

        async def script() -> dict[str, Any]:
            candidates: list[Candidate] = []
            best: Candidate | None = None
            best_idx: int | None = None
            calls = 0
            history: list[dict[str, Any]] = []

            while calls < self.max_calls:
                if best is not None and best.score >= self.solved_score:
                    break

                have = bool(candidates)
                action = self.policy.choose_action(
                    self._wider_arm, self._deeper_arm, have
                )
                if action == "deeper" and not have:
                    action = "wider"

                if action == "deeper" and best is not None:
                    arm = self._find_arm(best.model) or self.policy.choose_arm(
                        self.roster
                    )
                    prompt = REFINE_PROMPT.format(
                        problem=self.problem,
                        score=best.score,
                        answer=best.answer[:1500],
                        format_hint=self._format_hint,
                    )
                else:
                    action = "wider"
                    arm = self.policy.choose_arm(self.roster)
                    if best is not None and best.score < self.solved_score:
                        prompt = WIDEN_PROMPT.format(
                            score=best.score,
                            prev_answer=best.answer[:1500],
                            problem=self.problem,
                            format_hint=self._format_hint,
                        )
                    else:
                        prompt = SOLVE_PROMPT.format(
                            problem=self.problem, format_hint=self._format_hint
                        )

                failed = False
                try:
                    answer = (
                        await wf.agent(
                            prompt, model=arm.config, label=arm.config or "default"
                        )
                    ).strip()
                except Exception:
                    logger.warning(
                        "agent() raised for arm '%s'; recording as failed candidate",
                        arm.config,
                    )
                    answer = ""
                    failed = True

                # Score outside the guard: a model that raises is a tolerated
                # failed candidate, but a bug in the injected scorer is the
                # caller's and must surface, not be silently swallowed as
                # score 0.0. The scorer may be sync or async.
                if failed:
                    score = 0.0
                else:
                    raw = self.scorer(answer)
                    score = (await raw) if isinstance(raw, Awaitable) else raw

                calls += 1
                depth = 0 if action == "wider" else (best.depth + 1 if best else 0)
                candidate = Candidate(
                    answer=answer,
                    score=score,
                    model=arm.config,
                    action=action,
                    depth=depth,
                    parent=best_idx,
                )
                candidates.append(candidate)

                reward = 1.0 if best is None else max(0.0, min(1.0, score - best.score))
                if action == "wider":
                    self._wider_arm.alpha += reward
                    self._wider_arm.beta += 1.0 - reward
                else:
                    self._deeper_arm.alpha += reward
                    self._deeper_arm.beta += 1.0 - reward
                arm.alpha += reward
                arm.beta += 1.0 - reward

                if best is None or score > best.score:
                    best = candidate
                    best_idx = len(candidates) - 1

                history.append(
                    {
                        "call": calls,
                        "action": action,
                        "model": arm.config,
                        "score": score,
                        "failed": failed,
                    }
                )

            if best is None:
                return {
                    "final_answer": "",
                    "passed": False,
                    "attempts": calls,
                    "history": history,
                    "best_score": 0.0,
                }

            return {
                "final_answer": best.answer,
                "passed": best.score >= self.solved_score,
                "attempts": calls,
                "history": history,
                "best_score": best.score,
            }

        return await wf.workflow(script)


def create_ensemble_engine(
    problem: str,
    scorer: Scorer,
    models: list[str] | None = None,
    max_calls: int = 8,
    solved_score: float = 1.0,
    solution_format: str = "",
    policy: Any | None = None,
) -> EnsembleSearchEngine:
    """Wire an EnsembleSearchEngine with a roster of heterogeneous model arms.

    ``scorer(answer) -> float`` is the mandatory ground-truth ranker.
    ``models`` is a list of ``provider:model`` config strings; ``None``
    degenerates to a single-arm best-of-K loop on the session default.
    """
    if models is None:
        roster = [ModelArm(config=None)]
    else:
        roster = [ModelArm(config=m) for m in models]
    return EnsembleSearchEngine(
        problem=problem,
        scorer=scorer,
        roster=roster,
        max_calls=max_calls,
        solved_score=solved_score,
        solution_format=solution_format,
        policy=policy,
    )
