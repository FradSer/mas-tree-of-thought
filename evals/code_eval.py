"""Ground-truth code eval: engine vs baseline on HumanEval-style problems.

Verification executes the candidate implementation plus the problem's assert
tests in a subprocess with a timeout — pass/fail is objective, no LLM judge.
This runs model-generated code on the local machine, which is the standard
trade-off for HumanEval-style harnesses; problems are self-contained pure
functions and the subprocess gets a hard timeout.
"""

import asyncio
import re
import subprocess
import sys
import time
from collections.abc import Callable

from pydantic import BaseModel, Field

from .baseline import SingleCallBaseline
from .code_problems import CodeProblem
from .harness import EngineLike, count_agent_calls

# Code-focused discriminator rubric (the criteria steer answer content).
CODE_CRITERIA = """\
1. **Correctness**: Does the approach satisfy the specification and the
   docstring examples exactly, including edge cases (empty input, negatives,
   boundaries)?
2. **Completeness**: Is it a full, runnable implementation — no pseudocode,
   no missing helpers?
3. **Robustness**: Does it terminate on all inputs and avoid off-by-one and
   type errors?
4. **Simplicity**: Prefer a direct, readable implementation over a clever
   but fragile one."""

STATEMENT_TEMPLATE = """Implement the following Python function correctly.

```python
{prompt}```

Requirements:
- Satisfy the docstring specification and examples exactly, including edge cases.
- The implementation must be complete and runnable as-is.

Output: a single ```python code block containing the full function definition
(plus any helpers you need). No commentary outside the code block."""

_PYTHON_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
_ANY_BLOCK_RE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)


def build_statement(problem: CodeProblem) -> str:
    return STATEMENT_TEMPLATE.format(prompt=problem.prompt)


def extract_python_code(answer: str) -> str:
    """Extract the implementation from an answer.

    Prefers the last ```python block (final answers often restate the full
    implementation last), falls back to any fenced block, then to the raw
    text.
    """
    blocks = _PYTHON_BLOCK_RE.findall(answer) or _ANY_BLOCK_RE.findall(answer)
    if blocks:
        return blocks[-1].strip()
    return answer.strip()


class VerifyResult(BaseModel):
    """Outcome of executing one candidate against the problem's tests."""

    passed: bool
    output: str = Field("", description="Tail of stderr/stdout on failure.")


def verify_solution(
    problem: CodeProblem, code: str, timeout: float = 10.0
) -> VerifyResult:
    """Run ``code`` plus the problem's asserts in a subprocess."""
    program = f"{code}\n\n{problem.tests}"
    try:
        completed = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return VerifyResult(passed=False, output=f"timeout after {timeout}s")
    if completed.returncode == 0:
        return VerifyResult(passed=True)
    return VerifyResult(passed=False, output=(completed.stderr or "")[-500:])


class CodeProblemResult(BaseModel):
    """Engine-vs-baseline ground-truth outcome for one code problem."""

    problem_id: str
    engine_passed: bool
    baseline_passed: bool
    engine_error: str = ""
    baseline_error: str = ""
    engine_calls: int = Field(..., ge=0)
    baseline_calls: int = Field(..., ge=0)
    engine_seconds: float = Field(..., ge=0)
    baseline_seconds: float = Field(..., ge=0)


class CodeEvalReport(BaseModel):
    """All per-problem results plus pass-rate aggregates."""

    results: list[CodeProblemResult]
    engine_passed: int
    baseline_passed: int

    @classmethod
    def from_results(cls, results: list[CodeProblemResult]) -> "CodeEvalReport":
        return cls(
            results=results,
            engine_passed=sum(r.engine_passed for r in results),
            baseline_passed=sum(r.baseline_passed for r in results),
        )


async def run_code_eval(
    problems: list[CodeProblem],
    *,
    engine_factory: Callable[[str], EngineLike],
    baseline: SingleCallBaseline,
) -> CodeEvalReport:
    """Solve every problem with the engine and the baseline, verify both."""
    results: list[CodeProblemResult] = []
    for problem in problems:
        statement = build_statement(problem)

        with count_agent_calls() as engine_counter:
            engine_start = time.perf_counter()
            engine_run = await engine_factory(statement).run()
            engine_seconds = time.perf_counter() - engine_start
        engine_code = extract_python_code(engine_run["final_answer"])
        engine_verdict = await asyncio.to_thread(verify_solution, problem, engine_code)

        with count_agent_calls() as baseline_counter:
            baseline_start = time.perf_counter()
            baseline_answer = await baseline.answer(statement)
            baseline_seconds = time.perf_counter() - baseline_start
        baseline_code = extract_python_code(baseline_answer)
        baseline_verdict = await asyncio.to_thread(
            verify_solution, problem, baseline_code
        )

        results.append(
            CodeProblemResult(
                problem_id=problem.id,
                engine_passed=engine_verdict.passed,
                baseline_passed=baseline_verdict.passed,
                engine_error=engine_verdict.output,
                baseline_error=baseline_verdict.output,
                engine_calls=engine_counter.count,
                baseline_calls=baseline_counter.count,
                engine_seconds=engine_seconds,
                baseline_seconds=baseline_seconds,
            )
        )

    return CodeEvalReport.from_results(results)


class RescueProblemResult(BaseModel):
    """Engine outcome on one problem the baseline could not solve."""

    problem_id: str
    engine_passed: bool
    engine_error: str = ""
    engine_calls: int = Field(..., ge=0)
    engine_seconds: float = Field(..., ge=0)


class RescueReport(BaseModel):
    """Rescue-mode outcome: engine attempts only baseline failures.

    By construction the engine cannot regress problems the model already
    solves — it never touches them.
    """

    total_problems: int
    screen_attempts: int
    baseline_solved: list[str]
    attempted: list[RescueProblemResult]
    rescued: int


async def run_rescue_eval(
    problems: list,
    *,
    engine_factory: Callable[[str], EngineLike],
    baseline: SingleCallBaseline,
    screen_attempts: int = 2,
    verifier: Callable | None = None,
    statement_builder: Callable | None = None,
) -> RescueReport:
    """Screen with the baseline, then run the engine only on its failures.

    A problem counts as baseline-solved if any of ``screen_attempts``
    single-call attempts passes the tests; the engine is measured purely on
    its rescue rate over the remaining failures.

    ``verifier(problem, code) -> VerifyResult`` and
    ``statement_builder(problem) -> str`` default to the assert-style SWE
    suite; other suites (e.g. LiveCodeBench stdin problems) plug in theirs.
    """
    if verifier is None:
        verifier = lambda problem, code: verify_solution(problem, code)
    if statement_builder is None:
        statement_builder = build_statement

    solved: list[str] = []
    failures: list = []
    for problem in problems:
        statement = statement_builder(problem)
        passed = False
        for _ in range(screen_attempts):
            answer = await baseline.answer(statement)
            verdict = await asyncio.to_thread(
                verifier, problem, extract_python_code(answer)
            )
            if verdict.passed:
                passed = True
                break
        if passed:
            solved.append(problem.id)
        else:
            failures.append(problem)

    attempted: list[RescueProblemResult] = []
    for problem in failures:
        with count_agent_calls() as counter:
            start = time.perf_counter()
            engine_run = await engine_factory(statement_builder(problem)).run()
            seconds = time.perf_counter() - start
        code = extract_python_code(engine_run["final_answer"])
        verdict = await asyncio.to_thread(verifier, problem, code)
        attempted.append(
            RescueProblemResult(
                problem_id=problem.id,
                engine_passed=verdict.passed,
                engine_error=verdict.output,
                engine_calls=counter.count,
                engine_seconds=seconds,
            )
        )

    return RescueReport(
        total_problems=len(problems),
        screen_attempts=screen_attempts,
        baseline_solved=solved,
        attempted=attempted,
        rescued=sum(r.engine_passed for r in attempted),
    )


def render_rescue_markdown(report: RescueReport) -> str:
    """Render the rescue report as a Markdown summary."""
    failures = len(report.attempted)
    lines = [
        "# Dialectica rescue eval: engine on baseline failures (ground truth)",
        "",
        (
            f"**Baseline solved {len(report.baseline_solved)}/{report.total_problems} "
            f"(within {report.screen_attempts} attempts) · "
            f"Engine rescued {report.rescued}/{failures} of the failures**"
        ),
        "",
    ]
    if report.attempted:
        lines += [
            "| Failed problem | Engine | Engine calls | Engine s |",
            "|----------------|--------|--------------|----------|",
        ]
        for r in report.attempted:
            outcome = "RESCUED" if r.engine_passed else "still failing"
            lines.append(
                f"| {r.problem_id} | {outcome} | {r.engine_calls} "
                f"| {r.engine_seconds:.1f} |"
            )
        lines.append("")
    for r in report.attempted:
        if not r.engine_passed and r.engine_error:
            lines.append(f"## {r.problem_id} — engine failure")
            lines.append(f"```\n{r.engine_error}\n```")
            lines.append("")
    return "\n".join(lines)


def render_code_markdown(report: CodeEvalReport) -> str:
    """Render the code eval report as a Markdown summary."""
    total = len(report.results)
    lines = [
        "# Dialectica code eval: engine vs single-call baseline (ground truth)",
        "",
        (
            f"**Engine: {report.engine_passed}/{total} passed · "
            f"Baseline: {report.baseline_passed}/{total} passed**"
        ),
        "",
        "| Problem | Engine | Baseline | Engine calls | Engine s | Baseline s |",
        "|---------|--------|----------|--------------|----------|------------|",
    ]
    for r in report.results:
        engine = "pass" if r.engine_passed else "FAIL"
        baseline = "pass" if r.baseline_passed else "FAIL"
        lines.append(
            f"| {r.problem_id} | {engine} | {baseline} "
            f"| {r.engine_calls} | {r.engine_seconds:.1f} | {r.baseline_seconds:.1f} |"
        )
    lines.append("")
    for r in report.results:
        for arm, passed, error in (
            ("engine", r.engine_passed, r.engine_error),
            ("baseline", r.baseline_passed, r.baseline_error),
        ):
            if not passed and error:
                lines.append(f"## {r.problem_id} — {arm} failure")
                lines.append(f"```\n{error}\n```")
                lines.append("")
    return "\n".join(lines)


class AblationProblemResult(BaseModel):
    """Matched-cost ablation outcome for one problem."""

    problem_id: str
    engine_passed: bool
    best_of_n_passed: bool
    pass_at_1: bool
    n_budget: int = Field(..., ge=0, description="Engine calls = best-of-N budget.")
    best_of_n_first_pass: int | None = None
    engine_seconds: float = Field(0.0, ge=0)


class AblationReport(BaseModel):
    """Three-arm tally: pass@1 vs best-of-N vs full engine, at matched cost."""

    results: list[AblationProblemResult]
    engine_passed: int
    best_of_n_passed: int
    pass_at_1_passed: int

    @classmethod
    def from_results(cls, results: list[AblationProblemResult]) -> "AblationReport":
        return cls(
            results=results,
            engine_passed=sum(r.engine_passed for r in results),
            best_of_n_passed=sum(r.best_of_n_passed for r in results),
            pass_at_1_passed=sum(r.pass_at_1 for r in results),
        )


async def run_ablation(
    problems: list,
    *,
    engine_factory: Callable[[str], EngineLike],
    baseline: SingleCallBaseline,
    verifier: Callable | None = None,
    statement_builder: Callable | None = None,
    criteria_hint: str = "",
) -> AblationReport:
    """Compare the full engine against best-of-N single calls at matched cost.

    For each problem the engine runs first; its call count sets the budget N
    for the best-of-N arm (N independent single calls, pass if any passes).
    pass@1 is best-of-N's first sample. This isolates how much of the engine's
    lift is search/refinement versus plain resampling at the same spend.
    """
    if verifier is None:
        verifier = lambda problem, code: verify_solution(problem, code)
    if statement_builder is None:
        statement_builder = build_statement

    results: list[AblationProblemResult] = []
    for problem in problems:
        statement = statement_builder(problem)
        with count_agent_calls() as ec:
            start = time.perf_counter()
            engine_run = await engine_factory(statement).run()
            seconds = time.perf_counter() - start
        engine_code = extract_python_code(engine_run["final_answer"])
        engine_passed = (await asyncio.to_thread(verifier, problem, engine_code)).passed
        n = max(ec.count, 1)

        bo_statement = statement + criteria_hint
        first_pass = None
        for k in range(1, n + 1):
            answer = await baseline.answer(bo_statement)
            code = extract_python_code(answer)
            if (await asyncio.to_thread(verifier, problem, code)).passed:
                first_pass = k
                break

        results.append(
            AblationProblemResult(
                problem_id=problem.id,
                engine_passed=engine_passed,
                best_of_n_passed=first_pass is not None,
                pass_at_1=first_pass == 1,
                n_budget=n,
                best_of_n_first_pass=first_pass,
                engine_seconds=seconds,
            )
        )
    return AblationReport.from_results(results)


def render_ablation_markdown(report: AblationReport) -> str:
    """Render the ablation as a Markdown table."""
    n = len(report.results)
    lines = [
        "# Dialectica ablation: engine vs best-of-N at matched cost (ground truth)",
        "",
        (
            f"**Engine {report.engine_passed}/{n} · "
            f"best-of-N {report.best_of_n_passed}/{n} · "
            f"pass@1 {report.pass_at_1_passed}/{n}**"
        ),
        "",
        "| Problem | pass@1 | best-of-N | engine | N |",
        "|---------|--------|-----------|--------|---|",
    ]
    for r in report.results:
        p1 = "pass" if r.pass_at_1 else "FAIL"
        bon = f"pass@{r.best_of_n_first_pass}" if r.best_of_n_passed else "FAIL"
        eng = "pass" if r.engine_passed else "FAIL"
        lines.append(f"| {r.problem_id} | {p1} | {bon} | {eng} | {r.n_budget} |")
    return "\n".join(lines)
