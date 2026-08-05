# Dialectica ![](https://img.shields.io/badge/A%20FRAD%20PRODUCT-WIP-yellow)

[![PyPI](https://img.shields.io/pypi/v/dialectica.svg)](https://pypi.org/project/dialectica/) [![Twitter Follow](https://img.shields.io/twitter/follow/FradSer?style=social)](https://twitter.com/FradSer) [![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/) [![Framework](https://img.shields.io/badge/Framework-ADK%202.3+-orange.svg)]() [![Evaluation](https://img.shields.io/badge/Evaluation-honesty%20gate-purple.svg)]()

[English](README.md) | **简体中文**

**Dialectica** 是基于 Google ADK 的推理引擎工具箱，按硬方式构建与度量：每个引擎都跑过 matched-cost 基线和盲评判，只有数据支持的赢才保留，其余作为负向结果记录在案。全部目的就是用数字（而非感觉）回答一个问题——*scaffold 是否能打败一次精心提示的单次调用？*

> **一句话结论。** 在自包含任务上，纯 LLM scaffold（ToT、GAN、AB-MCTS scorer）*不*能在结果质量上打败 prompt-matched 单次调用——它们只是重排模型自己的思考。**辩证是唯一测出的例外**：把 synthesis 条调硬、螺旋加深后，它在开放式 meta-task 上确实打败 prompt-matched 单次调用（连续打分下从 **−0.500 到 +0.600 NET**，两次运行确认，见结论 #9）。引擎通过加入单次前向传播拿不到的信息来赢：**工具**、**ground-truth 验证**，或——在开放式 meta-task 上——**异构模型独立性**（实测：异构 reflection 在 10 题池上对单次 **10-0-0**；杠杆是 roster，不是 float scorer 或额外对抗 stage）。见[评测](#评测)。

受 [karpathy/autoresearch](https://github.com/karpathy/autoresearch)、Sakana AI 的 AB-MCTS / 集体智能系列、以及 Claude Code 可组合工作流启发。

## 公开 API（由数据支撑）

evals 把 ship 出去的接口收敛到数据真正支持的那一点：

| | 靠加入什么赢 | 判决 |
|---|---|---|
| **`Workflow` / `agent(tools=...)`** | **能力**——工具让一个 stage act → observe → iterate | ✅ 真赢（hidden-oracle 上 8/8 vs 0/8） |
| **`create_repair_engine`** | **ground truth**——验证器在环，通过即短路 | ✅ 成本赢（best-of-N 可靠性，约 1/3 调用） |

这个项目做过的其余东西——独立的 agentic 引擎类、异构 ensemble + scorer、辩证螺旋、
遗留 ToT+GAN beam search——要么用 `agent(tools=...)` 就够了，要么被测出作为纯 LLM
scaffold 与 prompt-matched 单次调用打平/输掉。它们被保留为可运行的**参考模式**，
不是 ship 出去的 API。开放式 meta-task 的实测配方是异构 reflection
（`examples/patterns/reflection_pattern.py`），组合在内核之上——仍不是第三个
ship 出去的引擎。见[模式](#模式不随包发布仅供参考)。

## 安装

```bash
uv add dialectica      # 或: pip install dialectica
```

```python
import os, asyncio
from dialectica import create_repair_engine

os.environ["GOOGLE_API_KEY"] = "..."  # 环境配置由应用负责


# 验证器对任意客观检查返回 (passed, feedback)——单元测试、JSON schema、
# linter、断言校验的业务逻辑。引擎据反馈反复修复，直到通过或用尽次数。
def verify(answer: str) -> tuple[bool, str]:
    ok = "def solve" in answer  # 你的真实检查写这里
    return ok, "" if ok else "no solve() function defined"


async def main():
    result = await create_repair_engine(
        "Write a solve() function that ...", verifier=verify
    ).run()
    print(result["passed"], result["attempts"], result["final_answer"])


asyncio.run(main())
```

可验证任务优先用 `create_repair_engine`。多步工具任务，构建一个 `Workflow`
脚本并直接调用 `agent(task, tools=[...])`（见下文）。库从 `os.environ` 读配置，
**不**自行加载 `.env`。

## Workflow 内核与 repair

### 🔗 `Workflow` / `agent` / `parallel` / `pipeline`——执行内核
可组合的多 agent 运行时——Claude Code `Workflow` 工具的**程序化**编排面（不含 IDE 宿主 UI）：
`agent()` / `parallel()` / `pipeline()` / `workflow()` / `phase()` / `log()` / `budget()` / `run_id()`。

- **`agent(..., isolation="worktree", agent_type="Explore", sees=[...])`**——`tools` 是能力加成杠杆；`isolation="worktree"` 在独立 git worktree 中运行（无变更则自动清理）；`agent_type` 应用预设角色章程；**`sees`** 是按步骤的访问列表（借鉴 Sakana Fugu-Ultra 的"防编排坍塌"机制）：默认完全隔离（一个 agent 永远看不到另一个 agent 的 transcript），`sees=["gather","critique"]` 只把指定前置步骤的输出注入本次调用的 prompt。未知/未完成的标签会被跳过而非报错，故访问列表能在条件分支下存活。
- **`workflow(script_or_name)`**——子 workflow 内联执行（一层嵌套），共享外层 budget；可用 `register_workflow` 按名称调用。
- **Resume**——`agent()` 调用记入 `.dialectica/workflows/<run_id>/`；`Workflow(..., resume_run_id=...)` 从缓存重放最长不变前缀。
- **护栏**——每 run 最多 1000 次 `agent()`；`parallel`/`pipeline` 每次最多 4096 项。
- **`Workflow(..., meta={...})`**——可选元数据；`phase()` 标题须与 `meta.phases` 一致。
- **诚实适用范围**：无 `tools` 的纯 schema 工作流仍是纯 LLM scaffold，受下方负向结论约束。

### 与 Claude Workflow 的对应——小模型能从这里得到什么

`Workflow` 内核是 Claude Code `Workflow` 工具的**程序化**编排面（不含 IDE `/workflows` UI）。
同样的 fan-out、分阶段 pipeline、子 workflow、resume、worktree 隔离，都可以用 Python 表达。

| Claude Code Workflow | Dialectica |
|---|---|
| `agent` / `parallel` / `pipeline` / `phase` / `log` / `budget` | ✅ |
| 子 workflow、`run_id`、resume/journal | ✅ |
| `agent(isolation="worktree")` | ✅ |
| `agent_type`（如只读 Explore） | ✅ 仅 Explore 预设 |
| 命名 workflow 注册表 | ✅ `register_workflow` |
| IDE `/workflows` UI、完整 agent 类型库（Plan 等） | ❌ 仅 API |
| 宿主深度集成（终端、文件树） | 自行注入 `tools` |

**这能让小参数模型「更强」吗？** 只有 workflow **加入了单次前向传播拿不到的信息** 时才行——与[评测](#评测)核心结论同一条定律。Workflow **形态本身不是智商放大器**。

| 场景 | 用法 | 小模型收益 |
|---|---|---|
| 必须读代码、跑命令、探测 API | `agent(tools=[...])`，可选 `parallel` | ✅ **实测真赢**——hidden-oracle 小模型 + tools **8/8**，单次调用 **0/8** |
| 输出可校验（测试、schema、linter） | `create_repair_engine` + verifier | ✅ **成本赢**——best-of-N 可靠性约 ⅓ 调用；通过率与 matched-cost 打平 |
| 开放式 meta-task（调研、评审、设计） | 异构 reflection：`create_reflection_engine`（或 `create_quality_workflow_engine(..., mode="reflection")`） | ✅ **实测赢**——异构 reflection 在 meta+default 上对单次 **10-0-0**（结论 #7）；杠杆是 roster 异构性 |
| 自包含推理（无工具、无 verifier） | 强单次 prompt 或更大模型 | ❌ 同模型 `phase`/`parallel` / 对抗 scaffold 打不过一次精心 prompt 的单次调用 |

**小模型实用配方：**

1. **探索 / 调试** — `agent_type="Explore"` + `tools=[...]`，可选 `isolation="worktree"`。
2. **可验证输出** — `create_repair_engine(verifier=...)`；失败时 `models=[小, 小, 中]` 轮换。
3. **调研 / 评审 / 开放式反思** — `create_reflection_engine(problem)` + 异构 roster（默认经 cliproxy 的 `qwen` + `glm`）。**不要**默认开 adversarial/dialectic——结论 #7 显示相对异构 reflection 无一致增益。
4. **控成本** — `Workflow(..., budget_unit="tokens")`；fan-out 用小模型，综合或最后一跳 repair 再用大模型。

`parallel` 与并发上限能降墙钟时间，不能抬高封闭式推理题的天花板。Context cache（见[配置](#配置)）在**单次 `agent()` 内的多轮 tool loop** 上省 token——独立 `agent()` 之间不会自动共享，除非自行管理 session。

### 🛠️ 执行制导修复——验证器在环（`create_repair_engine`）
可验证任务：**生成 → 跑注入的验证器 → 据具体失败修复 → 重试**，直到通过或
用尽次数。构建在 `Workflow` 内核之上——内部每次尝试都是循环里的一次
`agent(model=..., label=...)` 调用，不再有自己的一套 agent 构造逻辑。

- **任务无关的验证器**——任意 `Callable[[answer], (passed, feedback)]`：单测、
  schema 校验、linter、断言校验。`solution_format` 钉住验证器解析的输出形态。
- **用满失败历史**——每次失败尝试及其精确失败都回灌，避免在两个错误修复间
  振荡。
- **成本克制**——验证器一通过即短路，以远少于 best-of-N 的调用达到其可靠性。
- **多模型**——传 `models=[...]` 在失败时跨 roster 轮换；`history[i]["model"]`
  记录每次尝试由哪个模型产出。
- **返回** `{final_answer, passed, attempts, history}`。

实测判决（结论 #2）：通过率上打败单次调用、与 matched-cost best-of-K 打平，
但只需约 1/3 的调用。

## 模式（不随包发布，仅供参考）

`examples/patterns/`（与 `evals/` 一样是开发工具，不随 wheel 打包）保留了所有
被 evals 判定**不值得** ship 成稳定 API 的引擎的可运行参考实现。每个模式都保留
被降级引擎原本的工厂函数名/签名/返回形态，内部重建在 `Workflow` 内核之上而非
自建 agent。（当初测量它们的 `evals/*.py` 脚本已于 2026-08 清理中移除；实测判决
见上表与下方核心结论。）

| 模式 | 展示什么 | 实测判决 |
|---|---|---|
| `agentic_pattern.py`（`create_agentic_engine`） | `agent(tools=[...], instructions=...)` 作为独立的工具使用 stage | 与内核原语相同的 8/8 vs 0/8 胜绩——保留只是因为它是个带定制系统提示词的可直接复制的范例，不是因为这个能力需要一个类。 |
| `dialectic_pattern.py`（`create_dialectic_engine`） | 正 → 反 → 合螺旋，经 `agent(schema=Verdict)` 打分 | 自包含任务上对 prompt-matched 单次调用打平/输掉（**0-3-2**），但**调好后在开放式 meta-task 上打败单次调用**（调硬 synthesis + `max_rounds=5`：**−0.500 → +0.600 NET**，结论 #9）。 |
| `ensemble_pattern.py`（`create_ensemble_engine`） | 异构 roster 上的 AB-MCTS-lite 自适应搜索（Thompson 采样 bandit） | 被 honesty gate **CUT**——blind-pick roster（scorer 换成常数）打平了真实 scorer 的健壮性增益；信号相对异构性本身无额外贡献。 |
| `reflection_pattern.py`（`create_reflection_engine`） | **规范**开放式配方：异构 gather → frame → critique → synthesize，基于 `Workflow`。可选 `use_access_lists=True` 把 critique/synthesize 的前置上下文经由内核 `sees=` 原语注入（Fugu-Ultra 风格的选择性可见），而非用 `.format()` 内联。 | ✅ 实测赢——meta 上对单次/同构 **5-0-0**（#6）；经 quality ablation 在 meta+default 上对单次 **10-0-0**（#7）。无 LLM scorer / AB-MCTS。访问列表模式为可选项；上述实测数字用的是内联 prompt。 |
| `quality_workflow_pattern.py`（`create_quality_workflow_engine`） | 同一 roster 上的模式切换：`reflection`（默认，委托 reflection_pattern）/ `adversarial` / `dialectic` | Ablation 夹具——adversarial/dialectic 相对异构 reflection 无一致增益（#7）。除非比模式，否则优先 `create_reflection_engine`。 |
| `tot_gan_pattern.py`（`create_engine`/`create_coordinator`） | beam search + GAN 风格对抗精修，`parallel()` 用于兄弟展开/评估 | **实测被压制**——matched-cost 下从未赢过对单次/best-of-N/self-refine 的任何一场；在 24 点游戏上以约 34× 成本输给单次调用。 |

每个模式的 docstring 都写明其确切评测判决。它们按内核自身的组合风格编写
（对 `agent()`/`parallel()` 的纯函数/闭包组合），而非原来基于 Protocol 的插件
系统——保留是为了研究与历史数字的可复现性，不是为了扩展。像 evals 那样导入
它们：

```python
from examples.patterns.agentic_pattern import create_agentic_engine
from examples.patterns.dialectic_pattern import create_dialectic_engine
from examples.patterns.ensemble_pattern import create_ensemble_engine
from examples.patterns.reflection_pattern import create_reflection_engine
from examples.patterns.quality_workflow_pattern import create_quality_workflow_engine
from examples.patterns.tot_gan_pattern import create_engine
```

## 评测

引擎是否真的打败一次强模型调用？仓库附带评测工具（`evals/`，开发工具——不随
包发布），用数据回答：每题由引擎**和**单次调用基线各解一次；**盲评判**对两答案
各评两次并交换位置（不一致记 tie）；LLM 调用通过测试 mock 的同一 `run_agent`
接缝计数。

```bash
uv run python -m evals.reflection_ablation      # reflection 模式：异构 vs 同构 vs 单次（open-ended）
uv run python -m evals.quality_workflow_ablation  # 多模型模式 vs 单次（meta+default，10 题）
uv run python -m evals.workflow_ablation        # 同构 reflection vs 单次（open-ended）
```

历史评测脚本（ToT+GAN 的 `python -m evals` CLI，以及 `repair_ablation` /
`agentic_eval` / `quality_ablation` / `ensemble_ablation` /
`ensemble_meta_ablation` / `access_list_scale` / `scaffold_boundary` 套件）已于
2026-08 清理中移除；上面三个 ablation 是当前方法论，它们支撑的结论作为记录
保留在下方。

### 核心结论（实测，无预设结论）

1. **引擎真正赢的地方——能力，不是质量。** 在需要*行动*的任务上（agentic 隐藏
   oracle 基准），小模型用 `agent(tools=[...])` 得 **8/8**，单次调用 **0/8**：它探测
   隐藏函数、推断规则、实现之——单次调用无从知晓任意规则。这是真正的价值类别。

2. **scaffold 不赢的地方——自包含结果质量。** 与 *matched-cost* 基线相比，**无
   纯 LLM scaffold 打败单次调用**：dialectic 模式 vs prompt-matched 强基线在各档模型上
   **0-3-2**（早先 4-1-0 的"赢"是 prompt+长度，不是结构）。**repair** 引擎打败
   *单次*调用，但在通过率上与 *matched-cost best-of-K* **打平**——其真正优势是
   **成本**（best-of-N 可靠性，约 1/3 调用）。

3. **树结构被*压制*，而不仅无用。** 在 **24 点游戏**——ToT *自己*的标志基准
   上——忠实 ToT 得 **14/15，以约 34× 成本输给单次的 15/15**：现代模型一次解出
   2023 论文 GPT-4 失败 96% 的任务。matched-cost 盲评判下 ToT+GAN 模式
   **0-4-1 / 0-2-3 / 0-1-4**（vs 单次 / best-of-N / self-refine）——*从未赢过一
   场*。质量序：**self-refine ≥ best-of-N ≥ 单次 ≥ 树 scaffold**。

4. **价值窗口在可达模型范围上已关闭。** ToT 只在基模型单独失败但搜索能恢复的
   "失败但可修"区间有用。对最难的 24 点题在四个模型档位（最弱的可达云模型）上
   探测，单次调用**每个模型、每题都是 5/5**。没有可达的弱模型会失败这些任务，
   所以没有搜索可恢复的空隙——边界已越过此任务。

5. **异构 ensemble——scorer 的信号并非起作用者（2026-06-26）。** ensemble 设计
   为第四个诚实赢的杠杆——*独立性*由 ground-truth 级信号排序。两轴 honesty gate
   证伪了信号这一半的论题，同时浮现一个真实的更窄结果：
   - **代码（ground-truth 验证器，6 题，budget 6）：** ensemble+信号 **6/6**、
     best-single best-of-6 **6/6**、blind-pick **6/6**——**CUT**：两模型均一击解
     出，异构性与信号都无空间。饱和，同 #4。
   - **Open-ended meta（盲 LLM 评判，5 题，budget 6，位置交换）：** ensemble+信号
     以 **3-1-2** 打败 prompt-matched 单次调用——*模式确实在 open-ended 任务上
     提升回答健壮性*（代码轴测不出）。但 **blind-pick 臂**（信号换常数）也以
     **3-1** 打败单次：增益**归因于 roster 异构性，不是 scorer 排序信号**。按 H1
     信号归因条款：**CUT**。
   - **要点：** *无 scorer* 的多模型 best-of-N（采样 N 个异构模型、保留一个）即可
      捕获 ensemble 在 open-ended 上展现的健壮性增益；float scorer 相对 blind-pick
      无可测提升。repair 子判据亦 **CUT**（multi-model-repair@6 vs single@6：6/6 vs
      6/6，**0 次模型切换救援**）。

6. **异构 reflection——诚实的 meta-task 杠杆（2026-07-08）。** `reflection_pattern.py`
   实现 gather → frame → critique → synthesize 结构化 pipeline，各视角分配不同
   模型——无 AB-MCTS、无 LLM scorer。在完整 **5 题 meta 集**上（盲位置交换评判、
   cliproxy roster `openai:qwen3.6-flash` + `openai:glm-5.2`、
   `JUDGE_MODEL_CONFIG=openai:glm-5.2`、`DIALECTICA_DISABLE_THINKING=true`）：
   - **`evals/reflection_ablation.py`——异构 vs 同构 vs 单次：** 异构 reflection
     以 **5-0-0** 打败 prompt-matched 单次，以 **5-0-0** 打败同 pipeline 单模型
     ——增益**归因于 roster 异构性**，不只是多 stage 形状。
   - **`evals/workflow_ablation.py`——同构 vs 单次（对照）：** 同构 reflection pipeline
     以 **4-0-1** 打败单次（NET **+4**）——pipeline 形状在 meta-task 上*确实*有帮助，
     异构性补上剩余边际（含一题同构与单次 tie 但异构赢）。
   - **要点：** open-ended 反思/meta-task 用异构 multi-angle reflection；勿复活
     ensemble float-scorer 排序。复现：`uv run python -m evals.reflection_ablation`
     与 `uv run python -m evals.workflow_ablation`（cliproxy 环境同 #5）。

7. **多模型质量 workflow 模式——扩大题池（2026-07-09）。** `quality_workflow_pattern.py`
   在 **10 题**（5 meta + 5 default；盲评判、cliproxy roster 同 #6）上统一三种异构组合：
   - **vs 单次：** 同构 reflection **4-0-6**（NET +4）；异构 reflection **10-0-0**（NET +10）；
     异构 adversarial **9-0-1**（NET +9）；异构 dialectic **9-0-1**（NET +9）。
   - **vs 异构 reflection（额外 stage 是否有增益）：** adversarial **2-0-8**（NET +2）；
     dialectic **0-1-9**（NET −1）。
   - **要点：** 异构 `reflection` 为默认——扩大题池全胜。额外 adversarial-rival 或一轮
     dialectic 相对异构 reflection 无一致提升（多为 tie；dialectic 还输 1 场 head-to-head）。
     默认用 `create_reflection_engine`；`quality_workflow_pattern` 仅作模式对照。
     复现：`uv run python -m evals.quality_workflow_ablation`。

8. **访问列表——一种上下文可见性杠杆，移植自 Sakana Fugu（2026-07-12）。** 对 Sakana Fugu/Fugu-Ultra 编排器（TRINITY + The Conductor，ICLR 2026）的研究从另一侧印证了上述定律：Fugu 相对每个单 worker 的胜绩来自**模型独立性 + 学习型路由器**，而非 worker 缺少的工具，其机制是带按步骤访问列表的学习型通信拓扑。唯一能移植到无训练内核的机制是**访问列表**——`agent(sees=[...])` 现作为内核原语发布：默认完全隔离，可选注入指定前置步骤的输出。它经 `use_access_lists=True` 接入 reflection 配方（每个 critique 只看自己的 gather 角度；synthesize 只看 tension + critiques，而非全部 transcript），并已对照真实模型（经 OpenAI 兼容端点的 `glm-5.2`）验证。上述实测 reflection 数字用的是内联 prompt，故访问列表模式在 ablation 证明其在相同矩阵上 lift 或 tie 之前保持可选。复现实时校验：`uv run pytest -m e2e_access`。

9. **调好的辩证在开放式 meta-task 上打败 prompt-matched 单次调用（2026-08-05）。** 0-3-2（结论 #2）并不是全部：那个辩证没调到位。两处纯 LLM、同模型的改动——**调硬 `SYNTHESIS_PROMPT`**（做一个绑定决策、给出精确可测触发、说明每边赢的条件、保留具体数字——与 reflection 对 synthesis 的同一标准）和**加深螺旋**（`max_rounds` 3 → 5）——把辩证对 prompt-matched 强单次调用的 NET 从 **−0.500 翻到 +0.600**，经**两次独立运行**确认（+0.100、+0.600）。方法论很关键：这次用的是**连续 0-10 打分**（盲评判对每个答案按 `DEFAULT_CRITERIA` 打分，NET = 平均分差），而非离散胜/负/平——后者的 ±4 逐次摆动让早期测量不可读。被否的方向：再调硬 THESIS prompt（−0.333）和每轮两个对手（`perspectives=2`，−0.567）都回退并弃用。注意：在 3 个 meta 问题上、单一 judge（gpt-5.5）、连续打分设计下测得；同样的调法在完整 5-meta 题池上尚未实测。

### 这些结论共同指向的定律

scaffold 打败一次前向传播，**当且仅当**它加入了单次传播无法获得的信息——**工具**
（`agent(tools=...)`）、**ground-truth 验证**（repair），或**独立样本**（meta-task
上的异构模型——#6；经异构性的 ensemble 健壮性——#5）。对一个模型在单上下文上的纯
重排（ToT、GAN、辩证、对同族候选的 LLM-judge scorer）在自包含质量上与单次打平。
Sakana AI 的研究合集从另一侧收敛到同一定律：那里每个真正的赢也都由模型外部的
ground-truth oracle 支撑。这条定律正是为什么 ship 出去的 API 现在只剩两样东西——
能加能力的内核原语，和能加 ground truth 的那一个引擎——以及为什么其余全部搬去了
`examples/patterns/`（`reflection_pattern.py` 为经实测的 meta-task 参考）。

### 早期 advice 矩阵（2026-06-10/11）——已被取代

首轮矩阵将 ToT+GAN 模式与*较弱*的单次基线（无 prompt 匹配对照）和"Innovation"
判别准则（偏向过度复杂的答案）比较。已被上方 #2–#7 取代。记录于此：V1（Innovation
准则）技术上 7-1-1 赢、组织上 0-4-2 输；V2（Feasibility
准则）合计 20-8-2 vs V1 的 7-5-3——证明判别准则引导答案*内容*而非仅选择，但都未
打败 prompt-matched 强基线。

## 配置

所有配置从 `os.environ` 读取——作为库，Dialectica **不**自行加载 `.env`；环境
配置由消费应用负责。仅测试套件加载 `dialectica/.env`。

```bash
# 所有 agent 的默认模型
export DEFAULT_MODEL_CONFIG="google:gemini-3.5-flash"

# 角色特定覆盖（可选）——每次 wf.agent() 调用都用 Generator 角色
export GENERATOR_MODEL_CONFIG="google:gemini-3.5-flash"
# 供 evals/judge.py 的盲评判使用，与任何 ship 出去的引擎无关
export JUDGE_MODEL_CONFIG="google:gemini-3.1-pro-preview"

# Google AI Studio
export GOOGLE_API_KEY="..."

# 或 Vertex AI
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT="..."
export GOOGLE_CLOUD_LOCATION="..."

# OpenRouter
export OPENROUTER_API_KEY="..."

# OpenAI 兼容（proxy / vLLM / cliproxy）
export OPENAI_API_KEY="..."
export OPENAI_API_BASE="http://localhost:8317/v1"
# 关闭 qwen 族思考链以降评测延迟（可选）
export DIALECTICA_DISABLE_THINKING=true

# ADK 2.3+ 运行时（可选——见上文「与 Claude Workflow 的对应」）
export DIALECTICA_CONTEXT_CACHE=true              # 经 ADK App 开启 Gemini context cache
export DIALECTICA_CONTEXT_CACHE_MIN_TOKENS=4096   # Gemini 硬下限
export DIALECTICA_ADK_TELEMETRY=true              # 或改设 OTEL_EXPORTER_OTLP_*
```

只用 `gemini-3.5-flash`（默认）或 `gemini-3.1-pro-preview`——没有稳定的
`gemini-3.1-pro`（generateContent 返回 404）。provider 串为 `provider:model_name`；
`openai:` provider 显式传 `api_base`（新版 LiteLLM 不再为 `openai/` 前缀读
`OPENAI_API_BASE`）。

### 参数

- **`agent()`**——`tools`、`instructions`、`schema`、`model`（单次覆盖）、`isolation="worktree"`、`agent_type`（如 `"Explore"`）。
- **`Workflow`**——`budget_total` / `budget_unit`（`"calls"` 或 `"tokens"`）、`resume_run_id`、`meta`、`concurrency`；`budget().usage()` 在后端上报 cache hit 时含 `cached_tokens`。
- **`create_repair_engine`**——`verifier`（必填）、`max_attempts`、`solution_format`、`models`（可选 roster）。
- **模式**——见 `examples/patterns/` 里各模式自己的 docstring/工厂签名；它们保留了被降级引擎原本的参数（例如 ensemble 模式的 `scorer`/`policy`，dialectic 模式的 `criteria`/`rounds`）。

## 使用示例

### Repair（可验证任务）

```python
from dialectica import create_repair_engine


def verify(code: str) -> tuple[bool, str]:
    # 你的真实检查——跑测试、校验 schema 等
    return True, ""


engine = create_repair_engine("Write solve()", verifier=verify, max_attempts=3)
result = await engine.run()
# {"final_answer", "passed", "attempts", "history"}
```

### 工具使用 stage（内核原语）

```python
from dialectica import Workflow, agent


async def script():
    return await agent("Fix the failing test", tools=[read_file, run_tests])


result = await Workflow(script).run()  # 工具负责行动；事后检查结果
```

### 模式（仅供示意——不是 ship 出去的 API）

开放式 meta-task——规范异构 reflection（结论 #6 / #7）：

```python
from examples.patterns.reflection_pattern import create_reflection_engine

engine = create_reflection_engine(
    "Design the pricing tier",
    # 默认 roster：openai:qwen3.6-flash + openai:glm-5.2
    # use_access_lists=True  # 经 sees= 注入前置上下文，而非内联
)
result = await engine.run()
# {"final_answer", "history", "heterogeneous"}
```

Ensemble + float scorer 已被 **CUT**（结论 #5）——仅作历史 ablation；开放式质量
优先用上面的 reflection。

### 查看结果

`create_repair_engine` 和 `examples/patterns/` 里每个模式都返回含
`final_answer` 与轨迹（`history` / `attempts`）的 `dict`。repair 与 ensemble
模式的 `history` 记录每次尝试的产出模型；reflection 的 `history` 记录 stage、
label 与 model。

## 本地开发

```bash
uv sync                                         # 安装依赖
uv run pytest                                   # 模拟，快，无需 API key
uv run pytest -m e2e                            # 实时 E2E（需 GOOGLE_API_KEY）
uv run pytest -m e2e_access                     # 实时访问列表测试（需 OPENAI_API_BASE + OPENAI_API_KEY + DEFAULT_MODEL_CONFIG=openai:...）
uv run ruff format && uv run ruff check         # 格式化 / lint
```

库不调用 `logging.basicConfig`——日志配置由消费应用负责。在唯一接缝
`agent_runtime.run_agent()` 处 mock LLM——绝不 patch ADK 内部或各阶段 agent
（`tests/helpers.py` 有 fakes）。`asyncio_mode = auto`；pytest-bdd 步骤为同步，故
用 `asyncio.run()` 包协程。`examples/patterns/` 参考脚本只配一张更轻的回归网
（`tests/test_example_patterns_smoke.py`，每个模式跑一次端到端 mock）——完整的
BDD 场景覆盖只留给 ship 出去的内核 + repair。

## 测试流程（BDD 驱动 TDD）

新行为始于 `tests/features/*.feature` 中的 Gherkin 场景，经 pytest-bdd 执行——步骤
定义在 `tests/test_*_feature.py`（用 `scenarios(...)` 绑定）。然后 RED 测试 → GREEN
代码 → REFACTOR。更新测试时先改对应 `.feature`。CI
（`.github/workflows/test.yml`）在每次 push/PR 跑 `ruff format --check`、
`ruff check`、`pytest`。

## 项目结构

```
dialectica/
  adk_config.py        # ADK 2.3 context cache + OpenTelemetry 环境变量
  agent_factory.py    # 从 ROLE_TEMPLATES 构建 LlmAgent（只剩 Generator）
  agent_runtime.py    # 唯一 LLM 接缝：run_agent() + 重试/退避
  json_repair.py       # 共享的 fence/escape JSON 修复 helper
  llm_config.py         # provider:model 解析（google/openrouter/openai）
  repair.py             # create_repair_engine（成本赢）
  workflow.py           # Workflow + agent/parallel/pipeline/phase/log/budget + sees= 访问列表（内核）
examples/patterns/     # 被降级引擎的参考实现（不随包发布）
  agentic_pattern.py
  dialectic_pattern.py
  ensemble_pattern.py
  reflection_pattern.py       # 规范开放式配方（异构）；可选 use_access_lists
  quality_workflow_pattern.py # 模式 ablation 切换器
  tot_gan_pattern.py
evals/                # 仅开发的评测工具（不随 wheel 发布）
  baseline.py, harness.py, judge.py, problems.py, meta_problems.py  # 共享原语
  reflection_ablation.py, workflow_ablation.py, quality_workflow_ablation.py  # 当前 ablation
tests/                # BDD 特性 + 步骤定义 + helpers
```

## 故障排除

- **`gemini-3.1-pro` 返回 404**——用 `gemini-3.1-pro-preview` 或 `gemini-3.5-flash`。
- **OpenAI 兼容后端 "Connection error"**——新版 LiteLLM 不再为 `openai/` 前缀读
  `OPENAI_API_BASE`；库显式传 `api_base`，故确保设置了 `OPENAI_API_BASE`（而非仅
  `OPENAI_API_KEY`）。
- **qwen 族评测慢**——设 `DIALECTICA_DISABLE_THINKING=true` 关闭思考链
  （`chat_template_kwargs.enable_thinking=false`）。
- **Ensemble 模式 roster "collapsed to duplicate effective model"**（`examples/patterns/ensemble_pattern.py`）——这个模式降级后不再自动告警（该检查需要预构建的 agent，降级时被去掉了）；调用 `create_ensemble_engine` 前自己比对一下 `models` 列表有没有重复。
- **ToT+GAN 模式 + 强制 JSON 模式返回空判定**（部分后端，如 gemma-4-26b-a4b）——该模式的 `structured_output` 参数只为签名对齐保留，实际总是走 schema 强制打分；原引擎的绕过办法没有移植过来。

## 从 0.6.x 迁移

`create_agentic_engine`、`create_ensemble_engine`、`create_dialectic_engine`、
`create_engine`/`create_coordinator` 及其配套的 `Protocol`/模型类型**不再是公开
API 的一部分**。它们作为不随包发布的参考实现保留在 `examples/patterns/`
（`pip install dialectica` 不会装它们）：

```python
# 之前 (0.6.x)
from dialectica import create_agentic_engine

# 之后 (0.7.0)——签名/返回形态不变，现在是不随包发布的参考代码
from examples.patterns.agentic_pattern import create_agentic_engine
```

或者，对工具使用场景，直接用内核原语——不需要额外 import：

```python
from dialectica import Workflow, agent

result = await Workflow(lambda: agent(task, tools=[...])).run()
```

`create_repair_engine` 的签名与返回形态不变。`workflow.agent()` 新增了
`instructions=`（任务专属系统提示框架），现在能正确解析 `provider:model` 风格
的 `model=` 覆盖（此前未解析就直接透传），并新增 `sees=`（按步骤的访问列表，
用于选择性上下文可见性——受 Fugu-Ultra "防坍塌"启发；可选，默认行为不变）。
`dialectica.gan_evaluator` 改名为
`dialectica.json_repair`（只有共享的 fence/escape helper 保留下来；GAN 专属的类
迁去了 `examples/patterns/tot_gan_pattern.py`）。

## 贡献

约定式提交（用 `/git:commit` skill）。发布 = 推一个版本号与 `pyproject.toml`
**匹配**的 `v*.*.*` tag；CI 跑测试、发 PyPI、建 GitHub release。往公开 API 里加
东西时，连同 ship 会在数据说 CUT 时 CUT 它的 honesty-gate ablation——本仓的传统
是记录负向结果，而非未证实的声明。这次发布本身就是这套 honesty gate 的产物——正
是它把公开 API 收窄到只剩内核和 repair，见[模式](#模式不随包发布仅供参考)。

## 许可证

MIT——见 `LICENSE`。

## 参考资料

- [Tree of Thoughts](https://arxiv.org/abs/2305.10601)——Yao et al., 2023（ToT+GAN
  模式的谱系；现仅供参考）。
- [Sakana AB-MCTS / "Wider or Deeper?"](https://arxiv.org/abs/2503.04412)——ensemble
  模式的谱系（独立性 + ground-truth 信号）。
- [Sakana Fugu](https://sakana.ai/fugu/)——多模型协调器，其按步骤访问列表机制启发了内核 `sees=` 原语（结论 #8）。
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch)——灵感来源。

## 致谢

基于 Google ADK 构建。honesty-gate 方法得益于 LLM 评测中通用的盲位置交换评判模式。
