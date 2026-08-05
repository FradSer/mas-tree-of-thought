## Learned User Preferences

- For open-ended reflection and meta-tasks, prefer heterogeneous multi-angle workflows (`reflection_pattern.py`) over AB-MCTS ensembles or LLM-as-scorer ranking.
- Live multi-model evals should route through cliproxy via shell env (`CLIPROXYAPI_HOST`, `CLIPROXYAPI_TOKEN`); `evals/` scripts do not auto-load `dialectica/.env`.
- Set `DIALECTICA_DISABLE_THINKING=true` when running qwen-family models via cliproxy to reduce eval latency.
- Dialectica should mirror Claude Code Workflow orchestration semantics; nested scripts join outer budget via `wf.workflow()`, not isolated `Workflow().run()`.
- When executing attached implementation plans, do not edit the plan file; mark existing todos in_progress instead of recreating them.
- Reflect measured architectural insights into both `README.md` and `README.zh-CN.md` when asked.

## Learned Workspace Facts

- Shipped API is `Workflow` kernel + `create_repair_engine`; demoted engines live in `examples/patterns/` (agentic, dialectic, ensemble, reflection, quality_workflow, tot_gan).
- Dialectica thesis: pure-LLM scaffolds tie a prompt-matched single call *on self-contained tasks*; measured wins require tools, ground-truth verifiers, or heterogeneous model independence — AB-MCTS float scorer adds no lift over blind heterogeneity. Exception: the dialectic, correctly tuned (sharpened synthesis + `max_rounds=5`), beats a prompt-matched single call on open-ended meta-tasks (**−0.500 → +0.600 NET**, finding #9).
- Canonical open-ended recipe is `examples/patterns/reflection_pattern.py` (`create_reflection_engine`): Gather → Frame → Critique → Synthesize with per-angle model assignment; default roster `openai:qwen3.6-flash` + `openai:glm-5.2`. Measured **5-0-0** vs single/homo on meta; **10-0-0** vs single on meta+default.
- `quality_workflow_pattern.py` is an ablation mode switcher (`reflection` / `adversarial` / `dialectic`); finding #7: adversarial/dialectic add no consistent lift over hetero reflection — prefer `create_reflection_engine` unless comparing modes.
- `evals/reflection_ablation.py` compares hetero vs homo vs single on meta; `evals/workflow_ablation.py` is homogeneous reflection vs single; `evals/quality_workflow_ablation.py` compares modes on meta+default (10 problems).
- `workflow()` is exported from `dialectica.workflow` (not top-level `dialectica`) to avoid submodule name collision; patterns import `from dialectica import workflow as wf`.
- `wf.parallel()` expects a list of thunks, not a bare generator comprehension.
- Passing `get_model_config()`'s `LiteLlm` object into roster maps is fine for `wf.agent(model=...)`, but heterogeneity checks must normalize via a hashable key (see `reflection_pattern._model_key`).
- Cliproxy live eval defaults: `GENERATOR_MODEL_CONFIG=openai:qwen3.6-flash`, `JUDGE_MODEL_CONFIG=openai:glm-5.2` (or `qwen3.6-max-preview`), `DIALECTICA_WORKFLOW_CONCURRENCY=4`, `DIALECTICA_DISABLE_THINKING=true`.
- ADK 2.3 runtime toggles: `DIALECTICA_CONTEXT_CACHE=true` for multi-turn tool-loop caching within one `agent()` call; `DIALECTICA_ADK_TELEMETRY=true` or `OTEL_EXPORTER_OTLP_*` for OpenTelemetry; `TokenUsage.cached_tokens` surfaces cache hits.
- Workflow parity modules: `workflow_journal.py` (resume via `.dialectica/workflows/<run_id>/`), `workflow_registry.py` (`register_workflow`), `workflow_worktree.py` (`agent(isolation="worktree")`).
