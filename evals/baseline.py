"""Single-call baseline prompt for the live ablation scripts.

This is the control arm of the eval: the engine must beat what a single
well-prompted LLM call produces to justify its cost. Configure the model with
``BASELINE_MODEL_CONFIG`` (falls back to ``DEFAULT_MODEL_CONFIG``).
"""

BASELINE_INSTRUCTION = """Solve the following problem:

**Problem:**
{problem}

**Output:**
Provide the solution directly, without additional commentary."""
