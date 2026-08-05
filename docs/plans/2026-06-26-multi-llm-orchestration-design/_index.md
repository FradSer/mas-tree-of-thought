# Design: Heterogeneous Multi-LLM Orchestration

Status: design (committed for review). Date: 2026-06-26.

## Context

Dialectica's settled thesis (README "Evaluation", `CLAUDE.md`): no pure-LLM
scaffold beats a prompt-matched single call on self-contained tasks — scaffolds
only rearrange one model's own thinking on one context, adding no information
(dialectic **0-3-2**, ToT/GAN **0-4-1**). The two engines that *do* win add
something a single forward pass lacks: `agentic.py` (tool **capability**) and
`repair.py` (ground-truth **verification**).

Two research passes over Sakana AI's portfolio (orchestration; reflection/GAN)
converge on the same law from the other side: **every genuine Sakana win is
backed by a ground-truth oracle external to the model.** The transferable,
inference-time, API-level insight is that **heterogeneous models are a
new-information source** — different training distributions produce genuinely
independent candidates, the one thing a same-model scaffold can never
manufacture — **but heterogeneity only cashes out when paired with a real
selection signal.** This design adds that pairing as a fourth honest win lever
(**independence**, alongside capability, verification, scale), on probation
until the honesty gate (below) clears it.

The user's framing was "simulate Sakana's *entire* research." That is
deliberately rejected (see Glossary + scope boundary in Rationale): this is an
inference-time ADK library with **no weight access and no training loop**, so
weight-merging / weight-surgery / code-self-modification work is structurally
unbuildable here. We transfer exactly the two mechanisms that fit the domain and
match a confirmed win condition.

## Discovery Results

Codebase facts that shape the design (verified against source):

- **Single LLM seam**: `agent_runtime.run_agent(agent, instruction)`
  (`dialectica/agent_runtime.py:74`). The `agent` has its model baked in, so
  heterogeneity = holding N `LlmAgent`s; the seam needs no change and stays the
  one mock point.
- **Multi-provider config already exists**: `llm_config._parse_model_config`
  parses `"provider:model_name"` for google / openrouter / openai
  (`dialectica/llm_config.py:36-79`). It **silently falls back** to
  `gemini-3.5-flash` when a provider key is missing (`:53-79`) — a roster whose
  members all collapse to the same default is the worst failure mode (looks like
  an ensemble, is N copies of one model).
- **Agent construction**: `create_agent(role, role_name, model_config=...)`
  (`dialectica/agent_factory.py:66-116`) bakes the model into the `LlmAgent`.
- **Core engine to extend**: `repair.py` — `generate → verify → repair`, returns
  `{final_answer, passed, attempts, history}` (`dialectica/repair.py:78-116`);
  verifier is a mandatory positional `Callable[[str], tuple[bool, str]]` (`:33`).
- **Standalone-engine pattern**: `agentic.py` / `repair.py` are thin wirings that
  call `run_agent` directly. `workflow.py` is explicitly scoped to *meta-tasks
  without ground truth* (`dialectica/workflow.py:1-35`) — the wrong home for a
  ground-truth quality engine.
- **Eval methodology**: `evals/repair_ablation.py`, `evals/harness.py` —
  call-counting through the seam (`harness.py:30-48`), matched-cost vs best-of-N,
  blind position-swapped judge for meta-tasks.
- **Test seam**: fakes patch only `run_agent` and may dispatch on `agent.name`
  (`tests/helpers.py`); `agent.model` is a bare string only for the `google`
  provider (it is a `LiteLlm` object for others), so **key fakes on
  `agent.name`** and give each roster member a distinct stable name.

## Glossary

Canonical labels (reconciled across the three research sub-agents before
writing). Verification rule: `grep` across these four files must return only the
canonical label, never a rejected variant.

| Concept | Canonical label | Rejected / aliased variants |
|---|---|---|
| The declared set of N heterogeneous model configs | **roster** (`models: list[str]`) | "pool" (reads as worker/connection pool), "committee" (implies voting — we do not vote) |
| One roster member as a bandit decision unit | **arm** | — |
| Ground-truth ranking signal for the ensemble | **scorer** (`Scorer = Callable[[str], float]`, higher = better) | "value signal" (ok in prose), reusing "verifier" here |
| Repair's objective pass/fail checker | **verifier** (`Callable[[str], tuple[bool, str]]`, unchanged) | "scorer" |
| Eval-harness blind comparator | **judge** | reusing "scorer"/"verifier" |
| Generate a fresh candidate from a (possibly new) arm | **wider** | "explore", "branch" |
| Refine the current best candidate | **deeper** | "exploit", "iterate" |
| The wider-vs-deeper + which-arm decision component | **policy** (injectable; default = Thompson-sampling bandit) | "controller", "scheduler" |
| One generated answer under evaluation | **candidate** (`Candidate` dataclass) | "thought" (legacy ToT term — wrong shape) |
| The new engine | **`EnsembleSearchEngine`** / `create_ensemble_engine` | `EnsembleEngine`, `AdaptiveBranchingEngine`, "committee engine", "MultiLLM…" |
| Repair's per-attempt model attribution | **`history[i]["model"]`** | — |
| Per-attempt model-config list arg (both engines) | **`models`** | `model_pool` |
| The three-arm KEEP/CUT ablation | **honesty gate** | — |

Note on "explore/exploit": dropped entirely from code/prose to avoid clashing
with the legacy ToT engine's *Explore* phase. The user-facing and internal verbs
are both **wider/deeper** (Sakana's own vocabulary).

## Requirements

### Ensemble search engine (`dialectica/ensemble.py`)

- **FR1 (MUST)** Accept a **roster** of N≥1 heterogeneous `model_config` strings;
  each is an independently addressable **arm**. `models=None` → degenerate to
  `[get_model_config("GENERATOR")]` (single-model best-of-K).
- **FR2 (MUST)** Take a **mandatory injected scorer** `Callable[[str], float]` as
  a positional arg; refuse to construct without one (pure-scaffold misuse is
  unrepresentable).
- **FR3 (MUST)** Run AB-MCTS-lite adaptive search: each step decides **wider**
  (fresh candidate, possibly a new arm) vs **deeper** (refine current best) and
  which arm acts, via the injected **policy** (default: Beta-Bernoulli Thompson
  sampling). Round-robin is the documented degenerate fallback.
- **FR4 (MUST)** Return the top-scored candidate + an auditable trace: per call
  the action (wider/deeper), the producing model, and the score — mirroring
  repair's `{final_answer, passed, attempts, history}` contract.
- **FR5 (MUST)** Hard-bound by `max_calls: int`; stop early when
  `best.score >= solved_score` (default `1.0`).
- **FR6 (SHOULD)** Validate **roster distinctness** at construction — warn loudly
  (or error) if two arms resolve to the same effective model or fall back to the
  default (guards the "N copies of one model" trap).
- **FR7 (SHOULD)** Degrade cleanly to N=1 (strict superset of best-of-K under the
  same scorer).

### Multi-model repair (`dialectica/repair.py` extension)

- **FR8 (MUST)** Allow the generator to be a roster, rotating arm per attempt
  (failure → next arm), preserving the existing single-generator signature.
- **FR9 (MUST)** Keep the injected `verifier` as the selection signal unchanged —
  rotation changes *who generates*, never *how we judge*.
- **FR10 (MUST)** Record which model produced each attempt in `history` so the
  ablation can attribute fixes to switching vs same-model retry.
- **FR11 (SHOULD)** Round-robin default rotation; pluggable order later (no
  coupling to the ensemble bandit).

### Non-functional

- **NFR1 (MUST)** Back-compat: every existing single-model API stays byte-identical
  when the new `models` arg is omitted. Existing repair scenarios stay green.
- **NFR2 (MUST)** Task-agnostic signal: scorer/verifier injected as a plain
  `Callable`, never engine-internal LLM self-scoring.
- **NFR3 (MUST)** Cost discipline: all calls route through `run_agent` so
  `count_agent_calls()` measures them; bounded by `max_calls`.
- **NFR4 (MUST)** Offline-mockable at the single seam; no new patch points, no
  ADK-internal patching.
- **NFR5 (MUST)** Multi-provider via existing `llm_config` parsing; no
  provider-specific code in the engine.
- **NFR6 (MUST)** BDD-first; the honesty-gate ablation ships as a reproducible
  `evals/ensemble_ablation.py`, not an ad-hoc script.
- **NFR7 (SHOULD)** Concurrency-aware: any parallel seeding honors
  `DIALECTICA_MAX_CONCURRENCY` via the existing `run_agent` limiter.

## Rationale

### Why this, and the explicit scope boundary

**IN SCOPE** (each item adds a thesis-sanctioned win lever):

- **Heterogeneous ensemble search + mandatory scorer** — heterogeneity supplies
  new information (independent training distributions); the scorer supplies the
  real selection signal. Exactly the independence + ground-truth pairing.
- **Multi-model repair** — keeps repair's proven verifier and adds independence
  on the failure path at zero new selection-signal risk.
- **Three-arm honesty-gate ablation** — non-negotiable; the repo cuts features
  that only tie a single call, so the feature ships with its own falsifier.
- **Multi-provider roster wiring** — pure composition-root plumbing.

**OUT OF SCOPE** (out by domain and/or thesis):

- **Evolutionary model merging** (M2N2, CycleQD, ShinkaEvolve, LLM²) — needs
  weight access + a fitness loop over parameters. No weights here.
- **Weight-surgery self-adaptation** (Transformer²) — runtime weight modulation;
  no weight access at the API layer.
- **Code self-modification** (Darwin Gödel Machine) — needs a self-edit/training
  loop and is a documented reward-hacking hazard; adds no lever that injected
  tools (`agentic.py`) don't already provide more safely.
- **Game co-evolution demos** (Digital Red Queen) — a self-play research demo,
  not a reusable inference-time mechanism for a caller's task.
- **Any ensemble *without* a mandatory injected signal** — reduces to a
  same-information same-judge scaffold the thesis predicts ties a single call;
  permitted only as the honesty-gate blind-pick control, never shipped.
- **An LLM-judge baked into the engine as the signal** — same-model self-scoring
  adds nothing (measured); the signal must be caller-injected.

### Alternatives considered

- **Literal Sakana reproduction** — most of the portfolio is weight-space /
  training-time, physically out of reach; would be mechanism-cosplay with no
  thesis-sanctioned win.
- **A pure `workflow.py` script** — `agent(model=...)` already allows ad-hoc
  multi-model fan-out, but `workflow.py`'s own scope says it does not repeal the
  negative findings; a script there inherits "no signal → ties a single call." A
  first-class engine with a *mandatory* scorer encodes the one tie-breaking
  ingredient and makes it un-bypassable.
- **One configurable mega-engine** — violates "match complexity to scale."
  Ensemble search and multi-model repair are two separately-falsifiable
  mechanisms; fusing them hides which one earns its cost. Keep them separate,
  each behind its own gate, as repair/agentic/dialectic already are.

### Success criteria (the honesty gate)

The feature is **on probation** until `evals/ensemble_ablation.py` (style of
`repair_ablation.py`: route through the seam, count calls, ground truth where
available, blind position-swapped judge for meta-tasks) compares **three arms at
matched total cost**:

- **(a) ensemble + signal** — the feature.
- **(b) best-single best-of-N** — best-of-N on the strongest single roster
  member, same total call budget. The matched-cost resampling control.
- **(c) ensemble blind-pick** — the same N heterogeneous candidates, signal
  replaced by random pick. Isolates signal vs mere heterogeneity.

**KEEP** only if **(a) > (b)** on the primary metric at matched cost **AND
(a) > (c)** by a margin attributable to the signal. A cheaper *tie* with (b)
(repair-style short-circuit) is a secondary justification, never a substitute.
**CUT** (or demote to a documented negative result) if **(a) ≈ (b)** or
**(a) ≈ (c)**.

**Falsifiable hypothesis (H1):** *At matched total LLM-call cost, a heterogeneous
N-model ensemble ranked by a ground-truth-grade injected scorer achieves a
strictly higher pass-rate (verifiable) / net-win count (meta) than best-of-N on
the single strongest roster member, and the advantage collapses to within noise
when the signal is replaced by random selection.* **Corollary (H1-cost):** at
*equal* score the ensemble reaches it in fewer calls (early stop on a
high-scoring candidate) — secondary, not primary.

**Multi-model-repair sub-criterion:** on a verifiable set, multi-model-repair@K
must beat single-model-repair@K, with the FR10 `history` confirming wins came
from *model-switching*. If not, ship repair single-model only and drop the pool.

## Detailed Design

Full algorithm, data structures, signatures, and integration points are in
`architecture.md`. Executable Gherkin (happy path, cross-model rescue, mandatory
scorer, budget stop, all-reject best-effort, model-raises tolerance, repair
rotation, back-compat) is in `bdd-specs.md`. Cost/concurrency/determinism/honesty
pitfalls are in `best-practices.md`.

Headline shape:

```python
Scorer = Callable[[str], float]  # higher is better; mandatory, positional


def create_ensemble_engine(
    problem: str,
    scorer: Scorer,  # MANDATORY value signal
    models: list[str] | None = None,  # roster; None -> [get_model_config("GENERATOR")]
    max_calls: int = 8,
    solved_score: float = 1.0,
    solution_format: str = "",
    policy: Policy | None = None,  # default: Thompson bandit
) -> EnsembleSearchEngine: ...


# repair.py gains one optional kwarg, fully back-compatible:
def create_repair_engine(
    problem,
    verifier,
    max_attempts=3,
    model_config=None,
    solution_format="",
    models: list[str] | None = None,
): ...
```

`run()` returns `{final_answer, passed, attempts, history}` for both engines;
the ensemble's `history[i]` carries `{call, action, model, score}`, repair's
`history[i]` gains `model`.

## Design Documents

- `architecture.md` — system overview, AB-MCTS-lite algorithm, data structures,
  signatures, integration points, the new eval harness.
- `bdd-specs.md` — full Gherkin scenarios for both features.
- `best-practices.md` — security, performance, determinism, the honesty trap,
  and pitfalls.
