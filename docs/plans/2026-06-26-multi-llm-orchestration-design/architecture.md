# Architecture

Vocabulary is canonical per `_index.md` Glossary: **roster / arm / scorer /
verifier / wider / deeper / policy / candidate / EnsembleSearchEngine**.

## System overview

Two additions, both standalone (not built on `workflow.py`) and both routing
every LLM call through the single seam `agent_runtime.run_agent`:

1. **`dialectica/ensemble.py`** — `EnsembleSearchEngine` + `create_ensemble_engine`.
   AB-MCTS-lite adaptive search over a roster, ranked by a mandatory injected
   **scorer**. New file, mirroring the one-engine-per-file thin-wiring style of
   `repair.py` / `agentic.py`.
2. **`dialectica/repair.py` extension** — the generator may be a roster; rotates
   arm on each repair attempt. Additive, fully back-compatible.
3. **`evals/ensemble_ablation.py`** — the three-arm honesty gate.

**Why standalone, not on `workflow.py`** (`dialectica/workflow.py:1-35`): the
Workflow runtime is explicitly scoped to *meta-tasks without ground truth* and
documents that it does not repeal the negative findings. The ensemble is the
opposite category — a ground-truth-driven quality engine, like `repair.py`.
Building it on `Workflow` would misfile it, lose the per-engine `run()` dict the
evals consume, and gain nothing: `parallel`/`pipeline` are fan-out barriers,
whereas AB-MCTS is an inherently *sequential adaptive* loop (each step's decision
needs the prior step's score).

## The AB-MCTS-lite algorithm

Source: Sakana AB-MCTS ("Wider or Deeper?", arXiv 2503.04412; Multi-LLM
AB-MCTS). Standard MCTS only selects among existing children; AB-MCTS adds
**adaptive branching** — at every step a "generate a brand-new candidate" action
competes via **Thompson sampling** against "refine an existing one." Multi-LLM
variant: each model is a bandit arm; track which earns reward, Thompson-sample
the next. The reward is the **external scorer**, not an LLM self-score — which is
exactly Dialectica's thesis, so it transfers cleanly.

Faithful inference-time reduction (no training, no weights, no learned value
model, no rollouts — "AB-MCTS-lite"):

```
state:
  candidates: list[Candidate]          # flat, depth-bounded; root = problem
  best       = max(candidates, score)  # incumbent
  wider, deeper = BetaArm(), BetaArm() # action posteriors
  roster: list[ModelArm]               # each an arm with its own BetaArm

loop until best.score >= solved_score OR calls >= max_calls:
  1. action  = policy.choose_action(wider, deeper, have_candidates)
               # default: draw w~Beta(wider), d~Beta(deeper); argmax.
               # If no candidates yet, force WIDER (seed).
  2. arm     = policy.choose_arm(roster)
               # default: for each arm draw theta~Beta(arm); argmax (Thompson).
  3. answer  = one run_agent call:
        WIDER  -> run_agent(arm.agent, WIDEN_PROMPT)
                  # solve fresh; optionally inject the current best *failed*
                  # answer as an anti-hint ("a previous attempt scored X and
                  # was wrong: ...; produce a genuinely different approach").
                  # This is the cross-model information a single pass lacks.
        DEEPER -> run_agent(arm.agent, REFINE_PROMPT on best.answer + best.score)
                  # structurally identical to repair's REPAIR_PROMPT.
  4. score   = scorer(answer); append Candidate(answer, score, arm, action, depth)
  5. reward  = clip(score - best.score, 0.0, 1.0)   # continuous improvement
               # (first candidate: reward = 1.0). Avoids the 0/1 sparsity that
               # starves the bandit at small max_calls.
  6. update  Beta posteriors of the chosen action-arm and model-arm:
               alpha += reward; beta += (1 - reward)
     best    = max(best, new candidate)

return {
  "final_answer": best.answer,
  "passed":       best.score >= solved_score,
  "attempts":     calls,                 # == run_agent calls made
  "history": [{"call": i, "action": "wider"|"deeper",
               "model": cfg, "score": s}, ...],
  # optional: "best_score", "candidates"
}
```

The **policy** is an injectable seam (a small object/callable with
`choose_action` + `choose_arm`); the default is the Thompson bandit above. Tests
inject a scripted deterministic policy (`["wider","deeper","wider"]`) so the
schedule is asserted without RNG flakiness. The default bandit's own randomness
is seedable (`seed: int` or injected `random.Random`) for its focused unit test.

**Scorer.** `Scorer = Callable[[str], float]`, higher = better, **mandatory
positional**. A float (not pass/fail) is required because the engine *ranks*. The
`solved_score` threshold (default `1.0`) is the "objectively done" stop. A caller
with only a boolean verifier wraps it: `lambda a: 1.0 if v(a)[0] else 0.0`
(documented in the docstring). Feedback-aware scorers (`(float, str)`) are a
SHOULD/future extension; v1 refines "deeper" against the incumbent answer + its
score without requiring textual feedback.

## Data structures

Engine-internal `@dataclass`es (not Pydantic — never serialized across the LLM
seam; matches the "value object only where needed" judgment):

```python
@dataclass
class ModelArm:
    config: str  # "openai:qwen3.6", "google:gemini-3.5-flash"
    agent: LlmAgent  # built once via create_agent(model_config=config)
    name: str  # stable, distinct, e.g. "Candidate[gemini-3.5-flash]"
    alpha: float = 1.0  # Beta successes + 1
    beta: float = 1.0  # Beta failures + 1


@dataclass
class Candidate:
    answer: str
    score: float
    model: str  # the producing arm's config
    action: str  # "wider" | "deeper"
    depth: int
    parent: int | None  # index into candidates, or None for a wider seed
```

## The model-roster abstraction

The public boundary is a plain `list[str]` of `"provider:model"` configs — the
exact form `llm_config._parse_model_config` already parses
(`dialectica/llm_config.py:36-79`). 1–3 entries is the realistic scale (YAGNI:
no provider registry, no YAML, no sampler abstraction beyond Beta draws).

Construction:

- Build each arm's agent **once** via
  `create_agent(role="Generator", role_name=<distinct name>, model_config=cfg)`
  (`dialectica/agent_factory.py:66-116`). The model is baked into the `LlmAgent`,
  so heterogeneity = holding N agents; `run_agent` is unchanged and stays the
  single mock point.
- `models is None` → `[get_model_config("GENERATOR")]`
  (`dialectica/llm_config.py:82`) → degenerate single-model best-of-K.
- **FR6 distinctness check**: resolve each member's effective config; warn loudly
  (or raise) if two members resolve to the same effective model or fell back to
  the default (the silent-collapse trap in `llm_config.py:53-79`).

## Multi-model repair extension (back-compat)

Minimal additive change to `dialectica/repair.py`:

`IterativeRepairEngine.__init__` normalizes `generator` to a list:

```python
self._generators = generator if isinstance(generator, list) else [generator]
```

`run()` selects the generator per attempt by rotation (attempt 1 uses
`self._generators[0]`; with a one-element list this is byte-identical to today):

```python
gen = self._generators[(attempt - 1) % len(self._generators)]
```

`history[i]` gains a `model` field (the producing config), satisfying FR10. The
existing `{final_answer, passed, attempts, history}` keys are unchanged.

`create_repair_engine` gains one optional kwarg:

```python
def create_repair_engine(problem, verifier, max_attempts=3,
                         model_config=None, solution_format="",
                         models: list[str] | None = None):
```

- `models is None` → today's single-generator path, unchanged.
- `models` given → one Generator agent per config, passed as the list.
- `model_config` and `models` both given → `ValueError` (honest, one line; see
  the BDD "conflicting-config" scenario).

Relationship, to be cross-referenced in both docstrings: **ensemble** = adaptive
wider+deeper over a roster with a *float scorer*; **repair** = deeper-only
round-robin rotation with a *boolean verifier*. Repair is a degenerate ensemble.

## Public API

```python
# dialectica/ensemble.py
Scorer = Callable[[str], float]


class EnsembleSearchEngine:
    def __init__(
        self,
        problem,
        scorer,
        roster,
        max_calls=8,
        solved_score=1.0,
        solution_format="",
        policy=None,
    ): ...
    async def run(self) -> dict[str, Any]: ...


def create_ensemble_engine(
    problem: str,
    scorer: Scorer,  # MANDATORY, positional
    models: list[str] | None = None,
    max_calls: int = 8,
    solved_score: float = 1.0,
    solution_format: str = "",
    policy: "Policy | None" = None,
) -> EnsembleSearchEngine: ...
```

`__init__.py` (`dialectica/__init__.py:40-122`): add the imports, extend
`__all__` in the engine block, document `create_ensemble_engine` in the module
docstring hierarchy positioned as a *capability-adding* engine (ground-truth
scorer = information a single pass lacks → honest side of the thesis), and bump
`__version__`.

**No new Protocol.** The existing `Selector`/`Generator` protocols
(`dialectica/protocols.py:19-59`) are owned by the legacy ToT coordinator and
typed in `ThoughtData` terms — wrong vocabulary. The scorer (a plain `Callable`,
like `repair.Verifier` at `repair.py:33`) and the injectable policy are already
the swappable seams; YAGNI says don't abstract `Selector` over candidates yet.

## Integration points & risks

- **Single seam stays the only mock point.** All generation/refinement goes
  through `agent_runtime.run_agent` (`agent_runtime.py:74`); per-arm heterogeneity
  lives in the baked-in `LlmAgent.name`/`.model`, invisible to the seam, so a
  fake can key per-model responses off `agent.name`.
- **Concurrency.** AB-MCTS is sequential by nature (each decision needs the prior
  score) → low contention. If a wider seeding batch is ever parallelized, it
  inherits `DIALECTICA_MAX_CONCURRENCY` automatically via the limiter in
  `run_agent` (`agent_runtime.py:36-43, 94-96`). Cross-provider rate limits are
  already handled per-call (`agent_runtime.py:99-110`).
- **Structured output.** The scorer is an injected Python callable, not an LLM
  judge, so the engine sidesteps JSON-mode fragility (the gemma/qwen
  empty-verdict issues, `models.py:8-14`, `workflow.py:225-231`). Generation
  prompts stay plain text (no `output_schema`), like `repair.py`. A future
  LLM-scored variant should reuse `workflow._parse_structured` +
  `repair_json_escapes` (`workflow.py:256-294`), not re-roll parsing.
- **Eval harness** (`evals/ensemble_ablation.py`): mirror
  `evals/repair_ablation.py:101-139`. Three arms at matched call budget:
  (a) ensemble@`max_calls`, (b) best-of-`max_calls` on the strongest single
  roster member (pure wider, no adaptivity, no cross-hints — the resampling
  control), (c) ensemble blind-pick (signal replaced by random). Count calls via
  `count_agent_calls()` (`harness.py:30-48`) — unchanged because the engine uses
  the seam. Report scores **and** call counts. If (a) only ties (b), report it
  honestly, consistent with the dialectic/ToT negative results.
- **Risk — reward sparsity.** A coarse scorer + small `max_calls` updates
  posteriors slowly; the continuous improvement reward (step 5) mitigates, and
  round-robin is the documented floor (FR3). Keep the bandit simple — no learned
  value model (matches "match complexity to scale").
- **Risk — pool homogeneity in disguise.** If one model dominates, the ensemble
  degenerates toward best-of-N on it and H1 clause (a)>(b) fails; arm-level
  attribution in `history` surfaces this. Accept the negative result if it
  appears.
