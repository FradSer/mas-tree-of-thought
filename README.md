# Dialectica ![](https://img.shields.io/badge/A%20FRAD%20PRODUCT-WIP-yellow)

[![PyPI](https://img.shields.io/pypi/v/dialectica.svg)](https://pypi.org/project/dialectica/) [![Twitter Follow](https://img.shields.io/twitter/follow/FradSer?style=social)](https://twitter.com/FradSer) [![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/) [![Framework](https://img.shields.io/badge/Framework-ADK%202.3+-orange.svg)]() [![Evaluation](https://img.shields.io/badge/Evaluation-honesty%20gate-purple.svg)]()

**English** | [简体中文](README.zh-CN.md)

**Dialectica** is a reasoning-engine toolbox on Google ADK, built and measured the hard way: every engine is run against a matched-cost baseline and a blind judge, and only the wins the data supports are kept. The rest are documented as negative results. The whole point is to answer one question — *does a scaffold beat a single well-prompted call?* — with numbers, not vibes.

> **The one-sentence finding.** On self-contained tasks, a pure-LLM scaffold (ToT, GAN, AB-MCTS scorer) does *not* beat a prompt-matched single call on result quality — they only rearrange the model's own thinking. The **dialectic is the one measured exception**: with a sharpened synthesis bar and a deeper spiral, it *does* beat a prompt-matched single call on open-ended meta-tasks (from **−0.500** to **+0.600** NET, continuous score, confirmed over two runs — see finding #9). An engine wins by adding what one forward pass lacks: **tools**, **ground-truth verification**, or — on open-ended meta-tasks — **heterogeneous model independence** (measured: hetero reflection **10-0-0** vs single on a 10-problem pool; the lever is the roster, not a float scorer or extra adversarial stage). See [Evaluation](#evaluation).

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch), Sakana AI's AB-MCTS / collective-intelligence line, and Claude Code's composable workflows.

## The public API (data-justified)

The evals collapsed the shipped surface to exactly what the data supports:

| | Wins by adding | Verdict |
|---|---|---|
| **`Workflow` / `agent(tools=...)`** | **capability** — tools let a stage act → observe → iterate | ✅ genuine win (8/8 vs 0/8 on hidden-oracle) |
| **`create_repair_engine`** | **ground truth** — verifier-in-the-loop, short-circuits on pass | ✅ cost win (best-of-N reliability at ~1/3 the calls) |

Everything else this project built — a dedicated agentic-engine class, the heterogeneous ensemble + scorer, the dialectic spiral, the legacy ToT+GAN beam search — either needs nothing beyond `agent(tools=...)` or was measured to tie/lose a prompt-matched single call as a pure-LLM scaffold. They're kept as runnable **reference patterns**, not shipped API. For open-ended meta-tasks the measured recipe is hetero reflection (`examples/patterns/reflection_pattern.py`) composed on the kernel — still not a third shipped engine. See [Patterns](#patterns-not-shipped-for-reference).

## Install

```bash
uv add dialectica      # or: pip install dialectica
```

```python
import os, asyncio
from dialectica import create_repair_engine

os.environ["GOOGLE_API_KEY"] = "..."  # the app owns env setup


# A verifier returns (passed, feedback) for ANY objective check — unit tests,
# a JSON schema, a linter, assertion-checked logic. The engine repairs against
# the feedback until it passes or runs out of attempts.
def verify(answer: str) -> tuple[bool, str]:
    ok = "def solve" in answer  # your real check goes here
    return ok, "" if ok else "no solve() function defined"


async def main():
    result = await create_repair_engine(
        "Write a solve() function that ...", verifier=verify
    ).run()
    print(result["passed"], result["attempts"], result["final_answer"])


asyncio.run(main())
```

Prefer `create_repair_engine` for verifiable tasks. For multi-step tool-using
tasks, build a `Workflow` script and call `agent(task, tools=[...])` directly
(see below). The library reads configuration from `os.environ` and does
**not** load `.env` itself.

## Workflow kernel & repair

### 🔗 `Workflow` / `agent` / `parallel` / `pipeline` — the execution kernel
A composable multi-agent runtime — the programmatic surface Claude Code's
`Workflow` tool provides (IDE host UI excluded): `agent()` / `parallel()` /
`pipeline()` / `workflow()` / `phase()` / `log()` / `budget()` / `run_id()`.
For *meta-task* orchestration (research, review, planning, design).

- **`agent(prompt, *, schema=None, tools=None, instructions="", label=None, phase=None, model=None, isolation=None, agent_type=None, sees=None)`** — one LLM call. `schema` forces structured JSON; **`tools`** is the capability-add lever (8/8 vs 0/8 on hidden-oracle). `isolation="worktree"` runs in a fresh git worktree (auto-removed if clean). `agent_type` (e.g. `"Explore"`) applies a read-only exploration charter. ADK forbids `tools` + `schema` on one call — split across stages. **`sees`** is a per-step access list (inspired by Sakana Fugu-Ultra's anti-"orchestration-collapse" mechanism): default is full isolation (an agent never sees another agent's transcript); `sees=["gather","critique"]` injects only the named prior steps' outputs into this call's prompt. Unknown/unfinished labels are skipped, not errors, so access lists survive conditional branches.
- **`workflow(script_or_name, *, args=None)`** — inline child workflow (one nesting level); shares outer budget. Pass a registered name via `register_workflow`.
- **`parallel(thunks)`** / **`pipeline(items, *stages)`** — concurrent barrier / per-item staged flow; max 4,096 items per call; 1,000 `agent()` calls per run.
- **Resume** — each run journals `agent()` calls under `.dialectica/workflows/<run_id>/`; `Workflow(..., resume_run_id=...)` replays the longest unchanged prefix from cache.
- **`Workflow(..., meta={...})`** — optional `name`/`description`/`phases` metadata; phase titles must match `phase()` calls.
- **Honest scope**: schema-only judge/synthesize workflows (no `tools`) remain pure-LLM scaffolds bound by the negative findings below.

```python
from dialectica import Workflow
from dialectica import workflow as wf
from dialectica.workflow import register_workflow


async def research(args):
    wf.phase("Gather")
    return await wf.agent(f"Research: {args['topic']}")


register_workflow("research", research)
result = await Workflow(
    research,
    args={"topic": "cache design"},
    meta={
        "name": "research",
        "description": "fan-out research",
        "phases": [{"title": "Gather"}],
    },
).run()
```

### Claude Workflow parity — and what small models can gain from it

Dialectica's `Workflow` kernel is a **programmatic port** of Claude Code's
`Workflow` tool surface (IDE host UI excluded). You can express the same
orchestration patterns — fan-out, staged pipelines, child workflows, resume,
worktree isolation — as plain Python instead of a host-managed workflow file.

| Claude Code Workflow | Dialectica |
|---|---|
| `agent` / `parallel` / `pipeline` / `phase` / `log` / `budget` | ✅ |
| Child workflow, `run_id`, resume/journal | ✅ |
| `agent(isolation="worktree")` | ✅ |
| `agent_type` (e.g. read-only Explore) | ✅ Explore preset only |
| Named workflow registry | ✅ `register_workflow` |
| IDE `/workflows` UI, full agent-type roster (Plan, …) | ❌ API-only |
| Deep host integration (terminal, file tree) | Bring your own `tools` |

**Can this make a small model perform better?** Only when the workflow adds
information a single forward pass lacks — the same law as the [Evaluation](#evaluation)
headlines. Workflow *shape* is not an IQ amplifier.

| Situation | What to use | Small-model upside |
|---|---|---|
| Must read code, run commands, probe an API | `agent(tools=[...])`, optional `parallel` | ✅ **Measured win** — hidden-oracle **8/8 vs 0/8** for a small model with tools vs 0/8 single-call |
| Output is checkable (tests, schema, linter) | `create_repair_engine` + verifier | ✅ **Cost win** — best-of-N reliability at ~⅓ the calls; ties matched-cost pass-rate |
| Open-ended meta-task (research, review, design) | Hetero reflection: `create_reflection_engine` (or `create_quality_workflow_engine(..., mode="reflection")`) | ✅ **Measured win** — hetero reflection **10-0-0** vs single on meta+default (finding #7); lever is roster heterogeneity |
| Self-contained reasoning (no tools, no verifier) | Strong single prompt or bigger model | ❌ Same-model `phase`/`parallel` / adversarial scaffolds do not beat one well-prompted call |

**Practical recipe for small models:**

1. **Explore / debug** — `agent_type="Explore"` + `tools=[...]`, optionally `isolation="worktree"`.
2. **Verifiable output** — `create_repair_engine(verifier=...)`; rotate `models=[small, small, medium]` on failure.
3. **Research / review / open-ended reflection** — `create_reflection_engine(problem)` with a heterogeneous roster (default `qwen` + `glm` via cliproxy). Do **not** default to adversarial/dialectic modes — finding #7 found no consistent lift over hetero reflection.
4. **Cost control** — `Workflow(..., budget_unit="tokens")`; use a small model for fan-out, a larger one only for synthesis or the last repair attempt.

`parallel` + concurrency caps cut wall-clock; they do not raise a model's
reasoning ceiling on closed tasks. Context caching (below) saves tokens on
multi-turn **tool loops inside one `agent()` call** — not across independent
`agent()` stages unless you share session state yourself.

### 🛠️ Execution-guided repair — verifier-in-the-loop (`create_repair_engine`)
For verifiable tasks: **generate → run an injected verifier → repair against the
concrete failure → retry**, until it passes or attempts run out. Built on the
`Workflow` kernel — internally, each attempt is one `agent(model=..., label=...)`
call in a bounded retry loop, no bespoke agent construction of its own.

- **Task-agnostic verifier** — any `Callable[[answer], (passed, feedback)]`: unit tests, a schema validator, a linter, assertion-checked logic. `solution_format` pins the output shape your verifier parses.
- **Uses the full failure history** — every prior attempt + its exact failure is fed back, so the loop doesn't oscillate between two wrong fixes.
- **Cost-disciplined** — short-circuits the moment the verifier passes, reaching best-of-N reliability at a fraction of the calls.
- **Multi-model** — pass `models=[...]` to rotate across a roster on failure; the per-attempt `history[i]["model"]` records which model produced each attempt.
- **Returns** `{final_answer, passed, attempts, history}`.

Measure it against pass@1 and matched-cost best-of-K — the measured verdict
(finding #2): beats a single call on pass-rate, ties matched-cost best-of-K,
at ~1/3 the calls.

## Patterns (not shipped, for reference)

`examples/patterns/` (a dev tool, like `evals/` — not packaged in the wheel)
holds runnable reference implementations of everything the evals did **not**
justify shipping as stable API. Each keeps the demoted engine's exact
factory name/signature/return-shape, rebuilt on the `Workflow` kernel instead
of bespoke agent construction. (The `evals/*.py` scripts that originally
measured them were removed in the 2026-08 cleanup; their verdicts are recorded
in the table and the findings below.)

| Pattern | What it shows | Measured verdict |
|---|---|---|
| `agentic_pattern.py` (`create_agentic_engine`) | `agent(tools=[...], instructions=...)` as a standalone tool-using stage | Same 8/8 vs 0/8 win as the kernel primitive — kept only as a copy-pasteable recipe with the tailored system prompt, not because the capability needs a class. |
| `dialectic_pattern.py` (`create_dialectic_engine`) | thesis → antithesis → synthesis spiral, scored via `agent(schema=Verdict)` | Ties/loses a prompt-matched single call (**0-3-2**) on self-contained tasks, but **beats one on open-ended meta-tasks when tuned** (sharpened synthesis + `max_rounds=5`: **−0.500 → +0.600 NET**, finding #9). |
| `ensemble_pattern.py` (`create_ensemble_engine`) | AB-MCTS-lite adaptive search (Thompson-sampling bandit) over a heterogeneous roster | **CUT** by the honesty gate — a blind-pick roster (scorer replaced by a constant) matched the real scorer's robustness gain; the signal adds nothing over heterogeneity alone. |
| `reflection_pattern.py` (`create_reflection_engine`) | **Canonical** open-ended recipe: hetero gather → frame → critique → synthesize on `Workflow`. Opt-in `use_access_lists=True` routes critique/synthesize prior context through the kernel `sees=` primitive (Fugu-Ultra-style selective visibility) instead of inlining via `.format()`. | ✅ Measured win — **5-0-0** vs single/homo on meta (finding #6); **10-0-0** vs single on meta+default via quality ablation (finding #7). No LLM scorer / AB-MCTS. Access-list mode is opt-in; the measured numbers above used inlined prompts. |
| `quality_workflow_pattern.py` (`create_quality_workflow_engine`) | Mode switcher over the same roster: `reflection` (default, delegates to reflection_pattern) / `adversarial` / `dialectic` | Ablation harness — adversarial/dialectic add no consistent lift over hetero reflection (finding #7). Prefer `create_reflection_engine` unless comparing modes. |
| `tot_gan_pattern.py` (`create_engine`/`create_coordinator`) | beam search + GAN-style adversarial refinement, `parallel()` for sibling expand/evaluate | **Measured dominated** — never wins a matchup against single/best-of-N/self-refine at matched compute; loses to a single call on Game-of-24 at ~34× the cost. |

Each pattern's docstring cites its exact eval verdict. They're written in the
kernel's own compositional idiom (plain functions/closures over
`agent()`/`parallel()`), not the original Protocol-based plugin system —
kept for study and historical-number reproducibility, not for extension.
Import them the same way the evals do:

```python
from examples.patterns.agentic_pattern import create_agentic_engine
from examples.patterns.dialectic_pattern import create_dialectic_engine
from examples.patterns.ensemble_pattern import create_ensemble_engine
from examples.patterns.reflection_pattern import create_reflection_engine
from examples.patterns.quality_workflow_pattern import create_quality_workflow_engine
from examples.patterns.tot_gan_pattern import create_engine
```

## Evaluation

Does the engine actually beat a single strong-model call? The repo ships an
eval harness (`evals/`, a dev tool — not part of the published package) that
answers this with data: each problem is solved by the engine **and** by a
single-call baseline; a **blind judge** compares both answers twice with
positions swapped (disagreement = tie); LLM calls are counted through the same
`run_agent` seam the tests mock.

```bash
uv run python -m evals.reflection_ablation      # reflection pattern: hetero vs homo vs single (open-ended)
uv run python -m evals.quality_workflow_ablation  # multi-model modes vs single (meta+default, 10 problems)
uv run python -m evals.workflow_ablation        # homogeneous reflection vs single (open-ended)
```

The historical eval scripts (the ToT+GAN `python -m evals` CLI, and the
`repair_ablation` / `agentic_eval` / `quality_ablation` / `ensemble_ablation` /
`ensemble_meta_ablation` / `access_list_scale` / `scaffold_boundary` suites)
were removed in the 2026-08 cleanup; the three ablations above are the current
methodology, and the findings they back are retained below as a record.

### Headline findings (measured, no preset conclusion)

1. **Where an engine genuinely wins — capability, not quality.** On tasks that require *acting* (the agentic hidden-oracle benchmark), a small model with `agent(tools=[...])` scored **8/8** vs a single call's **0/8**: it probes the hidden function, infers the rule, and implements it — a single call can't know an arbitrary rule without probing. This is the genuine value class.

2. **Where scaffolds do NOT win — self-contained result quality.** Judged against a *matched-cost* baseline, **no pure-LLM scaffold beats a single call on self-contained tasks**: the dialectic pattern went **0-3-2** vs a prompt-matched strong baseline at every model size (the earlier 4-1-0 "win" was prompt + length, not structure). The **repair** engine beats a *single* call but exactly **ties matched-cost best-of-K** on pass-rate — its real edge is **cost** (best-of-N reliability at ~1/3 the calls). On *open-ended meta-tasks*, the picture differs — see finding #9 (the dialectic, correctly tuned, beats a prompt-matched single call).

3. **The tree structure is *dominated*, not just unhelpful.** On **Game-of-24** — ToT's *own* canonical benchmark — a faithful ToT scored **14/15 and lost to a single call's 15/15 at ~34× the cost**: modern models one-shot the task the 2023 paper's GPT-4 failed 96% of the time. At matched compute under a blind judge, the ToT+GAN pattern went **0-4-1 / 0-2-3 / 0-1-4** (vs single / best-of-N / self-refine) — it *never won a matchup*. The quality order is **self-refine ≥ best-of-N ≥ single ≥ tree-scaffold**.

4. **The value window is closed across the accessible model range.** ToT only helps where the base model fails alone but search can recover — a "fails-but-fixable" band. Probing the *hardest* Game-of-24 puzzles against **four model tiers** (the weakest cloud models available) a single call scored **5/5 on every model, every puzzle**. There is no accessible weak model that fails these tasks, so there is no gap for search to recover — the boundary has moved past this task.

5. **Heterogeneous ensemble — the scorer's signal is not what does the work (2026-06-26).** The ensemble was designed as a fourth honest win lever — *independence* ranked by a mandatory ground-truth-grade signal. A two-axis honesty gate falsified the signal half of the thesis while surfacing a real, narrower result:
   - **Code (ground-truth verifier, 6 problems, budget 6):** ensemble+signal **6/6**, best-single best-of-6 **6/6**, blind-pick **6/6** — **CUT**: both models one-shot every problem, so heterogeneity and the signal both have empty headroom. Saturation, same shape as finding #4.
   - **Open-ended meta (blind LLM-judge, 5 problems, budget 6, position-swap):** ensemble+signal beat a prompt-matched single call **3-1-2** — *the pattern does improve answer robustness on open-ended tasks* (the code axis couldn't measure this). But the **blind-pick arm** (signal replaced by a constant) also beat single **3-1**: the gain is **attributable to roster heterogeneity, not the scorer's ranking signal**. Per H1's signal-attribution clause: **CUT**.
   - **Takeaway:** a *no-scorer* multi-model best-of-N (sample N heterogeneous models, keep one) captures the robustness gain the ensemble shows on open-ended tasks; the float scorer adds no measurable lift over blind-pick. The repair sub-criterion was also **CUT** (multi-model-repair@6 vs single@6: 6/6 vs 6/6, **0 model-switch rescues**).

6. **Heterogeneous reflection — the honest meta-task lever (2026-07-08).** `reflection_pattern.py` implements the structured gather → frame → critique → synthesize pipeline with per-angle model assignment — no AB-MCTS, no LLM scorer. On the full **5-problem meta set** (blind position-swap judge, cliproxy roster `openai:qwen3.6-flash` + `openai:glm-5.2`, `JUDGE_MODEL_CONFIG=openai:glm-5.2`, `DIALECTICA_DISABLE_THINKING=true`):
   - **`evals/reflection_ablation.py` — hetero vs homo vs single:** heterogeneous reflection beat a prompt-matched single call **5-0-0** and beat the same pipeline on one model **5-0-0** — the gain is **attributable to roster heterogeneity**, not merely multi-stage shape.
   - **`evals/workflow_ablation.py` — homo vs single (control):** the homogeneous reflection pipeline beat single **4-0-1** (NET **+4**) — the pipeline shape *does* help on meta-tasks, but heterogeneity adds the remaining edge (including the one problem where homo tied single but hetero won).
   - **Takeaway:** for open-ended reflection/meta-tasks, use heterogeneous multi-angle reflection; do not resurrect ensemble float-scorer ranking. Reproduce: `uv run python -m evals.reflection_ablation` and `uv run python -m evals.workflow_ablation` (same cliproxy env as finding #5).

7. **Multi-model quality workflow modes — expanded pool (2026-07-09).** `quality_workflow_pattern.py` unifies three hetero compositions on **10 problems** (5 meta + 5 default; blind judge, same cliproxy roster as #6):
   - **vs single:** homo reflection **4-0-6** (NET +4); hetero reflection **10-0-0** (NET +10); hetero adversarial **9-0-1** (NET +9); hetero dialectic **9-0-1** (NET +9).
   - **vs hetero reflection (does the extra stage help?):** adversarial **2-0-8** (NET +2); dialectic **0-1-9** (NET −1).
   - **Takeaway:** hetero `reflection` is the default — it sweeps the expanded pool. Extra adversarial-rival or one-round dialectic stages add no consistent lift over hetero reflection (mostly ties; dialectic loses one head-to-head). Prefer `create_reflection_engine`; keep `quality_workflow_pattern` for mode comparison only. Reproduce: `uv run python -m evals.quality_workflow_ablation`.

8. **Access lists — a context-visibility lever, ported from Sakana Fugu (2026-07-12).** A study of Sakana's Fugu/Fugu-Ultra orchestrators (TRINITY + The Conductor, ICLR 2026) confirmed the law above from the other direction: Fugu's win over each single worker comes from **model independence + a learned router**, not tools the workers lack, and a learned communication topology with per-step access lists. The one mechanism portable to a no-training kernel is the **access list** — `agent(sees=[...])` now ships as a kernel primitive: default full isolation, opt-in to inject only designated prior steps' outputs. It is wired into the reflection recipe via `use_access_lists=True` (each critique sees only its own gather angle; synthesize sees the tension + critiques, not the full transcript), and verified against a live model (`glm-5.2` via an OpenAI-compatible endpoint). The measured reflection numbers above used inlined prompts, so access-list mode stays opt-in until an ablation shows it lifts or ties on the same matrices. Reproduce the live check: `uv run pytest -m e2e_access`.

9. **The dialectic, correctly tuned, beats a prompt-matched single call — on open-ended meta-tasks (2026-08-05).** The 0-3-2 result (finding #2) is **not the whole story**: that dialectic was under-tuned. Two pure-LLM, same-model changes — a **sharpened `SYNTHESIS_PROMPT`** (make ONE binding decision, give the precise measurable trigger, name the condition where each side wins, carry forward specific numbers — the same bar the reflection pattern holds its synthesis to) and a **deeper spiral** (`max_rounds` 3 → 5) — flip the dialectic from **−0.500 to +0.600 NET** vs a prompt-matched strong single call, confirmed over **two independent runs** (+0.100, +0.600). Methodology matters: this used a **continuous 0-10 score** (blind judge grades each answer against `DEFAULT_CRITERIA`, NET = mean score diff), not the discrete win/lose/tie NET whose ±4 run-to-run swing made the earlier measurement unreadable. Rejected directions: sharpening the THESIS prompt too (−0.333) and two rivals per round (`perspectives=2`, −0.567) both regressed and were reverted. Caveat: measured on the 3 meta problems, single judge (gpt-5.5), continuous-score design; the same tuning on the full 5-meta pool is not yet measured.

### The law these findings all point to

A scaffold beats one forward pass **iff** it adds information a single pass
lacks — **tools** (`agent(tools=...)`), **ground-truth verification** (repair), or
**independent samples** (heterogeneous models on meta-tasks — finding #6; ensemble
robustness via heterogeneity alone, finding #5). Pure rearrangement of one model's
thinking on one context (ToT, GAN, dialectic, an LLM-judge scorer over
same-family candidates) ties a single call on self-contained quality. Sakana AI's
portfolio converges on the same law from the other side: every genuine win there
is also backed by a ground-truth oracle external to the model. This law is
exactly why the shipped API is now two things — the kernel primitive that can add
capability, and the one engine that adds ground truth — and why everything else
moved to `examples/patterns/` (with `reflection_pattern.py` as the measured
meta-task reference).

### Earlier advice-suite matrices (2026-06-10/11) — superseded

The first-round matrices compared the ToT+GAN pattern against a *weaker*
single-call baseline (no matched-prompt control) and an "Innovation"
discriminator criterion that steered toward over-complex answers. They are
superseded by findings #2–#7 above. Recorded here: V1 (Innovation criterion) won technical problems 7-1-1 but lost organizational ones 0-4-2; V2 (Feasibility criterion) pooled to 20-8-2 vs V1's 7-5-3 —
evidence that discriminator criteria steer answer *content*, not just selection,
but neither beats a prompt-matched strong baseline.

## Configuration

All config is read from `os.environ` — as a library, Dialectica does **not**
load `.env` itself; the consuming app owns environment setup. Only the test
suite loads `dialectica/.env`.

```bash
# Default model for all agents
export DEFAULT_MODEL_CONFIG="google:gemini-3.5-flash"

# Role-specific override (optional) — every wf.agent() call uses the Generator role
export GENERATOR_MODEL_CONFIG="google:gemini-3.5-flash"
# Used by evals/judge.py's blind judge, not by any shipped engine
export JUDGE_MODEL_CONFIG="google:gemini-3.1-pro-preview"

# Google AI Studio
export GOOGLE_API_KEY="..."

# Or Vertex AI
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT="..."
export GOOGLE_CLOUD_LOCATION="..."

# OpenRouter
export OPENROUTER_API_KEY="..."

# OpenAI-compatible (proxy / vLLM / cliproxy)
export OPENAI_API_KEY="..."
export OPENAI_API_BASE="http://localhost:8317/v1"
# Disable qwen-family thinking trace for eval latency (optional)
export DIALECTICA_DISABLE_THINKING=true

# ADK 2.3+ runtime (optional — see "Claude Workflow parity" above)
export DIALECTICA_CONTEXT_CACHE=true              # Gemini context cache via ADK App
export DIALECTICA_CONTEXT_CACHE_MIN_TOKENS=4096   # Gemini hard floor
export DIALECTICA_ADK_TELEMETRY=true              # or set OTEL_EXPORTER_OTLP_* instead
```

Use `gemini-3.5-flash` (default) or `gemini-3.1-pro-preview` only — there is no
stable `gemini-3.1-pro` (404 on generateContent). Provider strings are
`provider:model_name`; the `openai:` provider passes `api_base` explicitly
(recent LiteLLM no longer reads `OPENAI_API_BASE` for the `openai/` prefix).

### Parameters

- **`agent()`** — `tools` (injected callables), `instructions` (task-specific guidance), `schema` (structured output), `model` (per-call override), `isolation="worktree"`, `agent_type` (e.g. `"Explore"`), `sees` (per-step access list of prior step labels for selective context visibility).
- **`Workflow`** — `budget_total` / `budget_unit` (`"calls"` or `"tokens"`), `resume_run_id`, `meta`, `concurrency`; `budget().usage()` includes `cached_tokens` when the backend reports cache hits.
- **`create_repair_engine`** — `verifier` (mandatory), `max_attempts`, `solution_format`, `models` (optional roster).
- **Patterns** — see each pattern's own docstring/factory signature in `examples/patterns/`; they keep their demoted engine's original parameters (e.g. `scorer`/`policy` for the ensemble pattern, `criteria`/`rounds` for the dialectic pattern).

## Usage examples

### Repair (verifiable task)

```python
from dialectica import create_repair_engine


def verify(code: str) -> tuple[bool, str]:
    # your real check — run the tests, validate the schema, etc.
    return True, ""


engine = create_repair_engine("Write solve()", verifier=verify, max_attempts=3)
result = await engine.run()
# {"final_answer", "passed", "attempts", "history"}
```

### Tool-using stage (kernel primitive)

```python
from dialectica import Workflow, agent


async def script():
    return await agent("Fix the failing test", tools=[read_file, run_tests])


result = await Workflow(script).run()  # tools do the acting; check the outcome after
```

### Patterns (illustrative only — not shipped API)

Open-ended meta-tasks — canonical hetero reflection (finding #6 / #7):

```python
from examples.patterns.reflection_pattern import create_reflection_engine

engine = create_reflection_engine(
    "Design the pricing tier",
    # default roster: openai:qwen3.6-flash + openai:glm-5.2
    # use_access_lists=True  # route prior context through sees= instead of inlining
)
result = await engine.run()
# {"final_answer", "history", "heterogeneous"}
```

Ensemble + float scorer is **CUT** (finding #5) — kept only for historical
ablation; prefer reflection above for open-ended quality.

### Inspecting the result

`create_repair_engine` and every pattern in `examples/patterns/` return a
`dict` with `final_answer` plus a trace (`history` / `attempts`). Repair and
ensemble-pattern `history` entries carry the producing model per attempt;
reflection `history` records stage, label, and model.

## Development

```bash
uv sync                                         # install deps
uv run pytest                                   # mocked, fast, no API key
uv run pytest -m e2e                            # live E2E (needs GOOGLE_API_KEY)
uv run pytest -m e2e_access                     # live access-list tests (needs OPENAI_API_BASE + OPENAI_API_KEY + DEFAULT_MODEL_CONFIG=openai:...)
uv run ruff format && uv run ruff check         # format / lint
```

The library never calls `logging.basicConfig` — the consuming app owns logging.
Mock the LLM at the single seam `agent_runtime.run_agent()` — never patch ADK
internals or per-stage agents (`tests/helpers.py` has the fakes). `asyncio_mode =
auto`; pytest-bdd steps are sync, so wrap coroutines with `asyncio.run()`.
The `examples/patterns/` reference scripts get a lighter regression net
(`tests/test_example_patterns_smoke.py`, one mocked end-to-end run each) —
full BDD scenario coverage is reserved for the shipped kernel + repair.

## Testing workflow (BDD-driven TDD)

New behavior starts with a Gherkin scenario in `tests/features/*.feature`,
executable via pytest-bdd — step definitions live in `tests/test_*_feature.py`
(bound with `scenarios(...`). Then RED test → GREEN code → REFACTOR. When
updating tests, update the matching `.feature` first. CI
(`.github/workflows/test.yml`) runs `ruff format --check`, `ruff check`, and
`pytest` on every push/PR.

## Project structure

```
dialectica/
  adk_config.py        # ADK 2.3 context cache + OpenTelemetry env wiring
  agent_factory.py    # builds LlmAgents from ROLE_TEMPLATES (Generator only)
  agent_runtime.py    # THE single LLM seam: run_agent() + retry/backoff
  json_repair.py       # shared fence/escape JSON-repair helpers
  llm_config.py        # provider:model parsing (google/openrouter/openai)
  repair.py            # create_repair_engine (cost win)
  workflow.py           # Workflow + agent/parallel/pipeline/phase/log/budget + sees= access lists (the kernel)
examples/patterns/     # reference implementations of demoted engines (not shipped)
  agentic_pattern.py
  dialectic_pattern.py
  ensemble_pattern.py
  reflection_pattern.py       # canonical open-ended recipe (hetero); opt-in use_access_lists
  quality_workflow_pattern.py # mode ablation switcher
  tot_gan_pattern.py
evals/                # dev-only eval harness (not shipped in the wheel)
  baseline.py, harness.py, judge.py, problems.py, meta_problems.py  # shared primitives
  reflection_ablation.py, workflow_ablation.py, quality_workflow_ablation.py  # the current ablations
tests/                # BDD features + step defs + helpers
```

## Troubleshooting

- **`gemini-3.1-pro` 404s** — use `gemini-3.1-pro-preview` or `gemini-3.5-flash`.
- **OpenAI-compatible backend "Connection error"** — `OPENAI_API_BASE` is no longer read for the `openai/` prefix by recent LiteLLM; the library passes `api_base` explicitly, so make sure `OPENAI_API_BASE` is set (not just `OPENAI_API_KEY`).
- **qwen-family evals are slow** — set `DIALECTICA_DISABLE_THINKING=true` to disable the reasoning trace (`chat_template_kwargs.enable_thinking=false`).
- **Ensemble pattern roster "collapsed to duplicate effective model"** (`examples/patterns/ensemble_pattern.py`) — this pattern no longer warns automatically (the check needed pre-built agents, dropped when it was demoted); compare your `models` list for duplicates yourself before calling `create_ensemble_engine`.
- **ToT+GAN pattern + enforced JSON mode returns empty verdicts** (some backends, e.g. gemma-4-26b-a4b) — the pattern's `structured_output` parameter is accepted for signature parity but always uses schema-enforced scoring; the original engine's workaround was not ported.

## Migration from 0.6.x

`create_agentic_engine`, `create_ensemble_engine`, `create_dialectic_engine`,
`create_engine`/`create_coordinator` and their supporting `Protocol`/model
types are **no longer part of the public API**. They remain available as
unshipped reference implementations in `examples/patterns/` (not installed
via `pip install dialectica`):

```python
# before (0.6.x)
from dialectica import create_agentic_engine

# after (0.7.0) — same signature/return-shape, now unshipped reference code
from examples.patterns.agentic_pattern import create_agentic_engine
```

Or, for the tool-using case specifically, use the kernel primitive directly —
no separate import needed:

```python
from dialectica import Workflow, agent

result = await Workflow(lambda: agent(task, tools=[...])).run()
```

`create_repair_engine`'s signature and return shape are unchanged.
`workflow.agent()` gained `instructions=` (task-specific system-prompt
framing), now correctly resolves `provider:model`-style `model=` overrides
(previously passed through unresolved), and gained `sees=` (a per-step
access list for selective context visibility — Fugu-Ultra-inspired
anti-collapse; opt-in, default unchanged). `dialectica.gan_evaluator`
is renamed `dialectica.json_repair` (only the shared fence/escape helpers
survive; the GAN-specific classes moved to `examples/patterns/tot_gan_pattern.py`).

## Contributing

Conventional commits (use the `/git:commit` skill). Release = push a `v*.*.*`
tag whose version **matches** `pyproject.toml`; CI runs tests, publishes to PyPI,
and creates a GitHub release. When adding to the shipped API, ship the
honesty-gate ablation that would CUT it if the data says so — the repo's
tradition is documented negative results, not unproven claims. This release's
own honesty gate is the reason the shipped surface is now just the kernel and
repair — see [Patterns](#patterns-not-shipped-for-reference).

## License

MIT — see `LICENSE`.

## References

- [Tree of Thoughts](https://arxiv.org/abs/2305.10601) — Yao et al., 2023 (the ToT+GAN pattern's lineage; now reference-only).
- [Sakana AB-MCTS / "Wider or Deeper?"](https://arxiv.org/abs/2503.04412) — the ensemble pattern's lineage (independence + ground-truth signal).
- [Sakana Fugu](https://sakana.ai/fugu/) — the multi-model coordinator whose per-step access-list mechanism inspired the kernel's `sees=` primitive (finding #8).
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — inspiration.

## Acknowledgments

Built on Google ADK. The honesty-gate methodology owes to the blind
position-swapped judge pattern used across LLM evals.
