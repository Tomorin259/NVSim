# NVSim

## English

### Overview

NVSim is a lightweight GRN-aware RNA velocity simulator.

Its current public scope is a deterministic benchmark scaffold built around a transparent modeling chain:

```text
GRN -> alpha(t) -> unspliced/spliced ODE -> true velocity -> snapshot cells -> observed layers
```

NVSim is intentionally smaller than SERGIO, VeloSim, dyngen, or scVelo-style pipelines. The goal is a readable simulator with inspectable ground truth rather than a full biological simulator.

### What NVSim Implements

- Explicit GRN input with `regulator`, `target`, `sign`, `K`, `half_response`, and `hill_coefficient`.
- Explicit or inferred master regulators.
- SERGIO-style additive Hill-function production rates for non-master genes.
- Configurable regulator activity for Hill regulation: `spliced` (default), `unspliced`, or `total = u + s`.
- Two master-regulator alpha source modes:
  `continuous_program` for NVSim-style continuous alpha programs, and
  `state_anchor` for SERGIO-style state/bin production anchors.
- Deterministic RNA velocity ODEs for unspliced and spliced RNA.
- Separate true and observed layers.
- Graph-based state-trajectory simulation over rooted DAGs, including path-like and branching templates.
- Optional AnnData export and quick-look plotting utilities.

### What NVSim Does Not Implement

- SERGIO CLE/SDE stochastic simulation.
- Promoter switching.
- Normal/MURK/branching gene classes.
- VeloSim EVF-to-kinetics mapping.
- Full scVelo-style velocity embedding.

### Core Equations

For gene `i`, NVSim uses:

```text
du_i/dt = alpha_i(t) - beta_i * u_i(t)
ds_i/dt = beta_i * u_i(t) - gamma_i * s_i(t)
v_i(t) = beta_i * u_i(t) - gamma_i * s_i(t)
```

Here:
- `alpha_i(t)` is the transcription / production rate.
- `beta_i` is the splicing rate.
- `gamma_i` is the degradation rate.
- `true_velocity` is `ds/dt`.
- `true_velocity_u` is `du/dt`.

### Source Map

The public API follows a flat `nvsim/*.py` layout so the core model is easy to inspect.

- `nvsim/grn.py`: GRN schema, master-regulator detection, graph levels, and half-response calibration helpers.
- `nvsim/regulation.py`: SERGIO-style Hill activation/repression and additive target production.
- `nvsim/production.py`: master-regulator forcing definitions, including time-dependent alpha programs and state/bin-wise production profiles.
- `nvsim/simulate.py`: beta/gamma setup, initial `u0/s0` validation, deterministic ODE integration, snapshot sampling, and result assembly.
- `nvsim/noise.py`: observed-count generation for canonical `poisson_capture` and `binomial_capture` models.
- `nvsim/output.py`: plain dict output and optional AnnData export.
- `nvsim/plotting.py`: scanpy/scVelo-based velocity showcase workflow plus mechanistic diagnostics such as gene dynamics and phase portraits.
- `nvsim/sergio_io.py`: SERGIO `targets/regs` parser.

### Install

Python 3.10 or newer is required.

```bash
pip install -e .
```

Development install:

```bash
pip install -e .[dev]
```

scanpy and scVelo are installed as core dependencies because the public plotting workflow delegates PCA/neighbors/UMAP to scanpy and velocity stream visualization to scVelo.

### Public API Walkthrough

#### 1. Define GRN

Define a GRN as a DataFrame with canonical columns:

- `regulator`
- `target`
- `sign`
- `K`
- `half_response`
- `hill_coefficient`

Minimal example:

```python
import pandas as pd

from nvsim.grn import GRN

edges = pd.DataFrame(
    {
        "regulator": ["g0"],
        "target": ["g1"],
        "sign": ["activation"],
        "K": [0.8],
        "half_response": [0.5],
        "hill_coefficient": [2.0],
    }
)

grn = GRN.from_dataframe(edges, genes=["g0", "g1"])
```

Use `sign="activation"` / `"repression"` for readability. `+` / `-` are also accepted at input time.

#### 2. Choose Alpha Source

NVSim supports two master-regulator alpha source modes:

- `continuous_program`: the original NVSim mode. Master-regulator alpha is a time function, `alpha_m(t) = f_m(t)`. Use this for clean, controlled pseudotime benchmarks.
- `state_anchor`: a SERGIO-inspired mode. Each state/bin/cell type has a master-regulator production vector, and transitions can interpolate from a parent state to a child state with `step`, `linear`, or `sigmoid` schedules.

`StateProductionProfile` values are user-supplied simulation design inputs. They can be hand-written, sampled, copied from SERGIO-style production tables, or estimated from external cluster-level TF activity.

For graph-based `state_anchor` simulation, provide a graph plus a production profile whose index matches graph states. Each edge defines a parent-to-child transition, and `transition_schedule` controls how source production moves along each segment. `transition_schedule="sigmoid"` is usually the best default for differentiation-like trajectories because it avoids hard alpha jumps.

Regulatory contributions can choose which RNA state is treated as regulator activity:

- Recommended default: `regulator_activity="spliced"`
- For SERGIO-style dynamic comparison: `regulator_activity="unspliced"`
- For sensitivity analysis: compare `"spliced"`, `"unspliced"`, and `"total"`

If a `StateProductionProfile` is available, half-response calibration can run inside simulation:

- `auto_calibrate_half_response=False` (default): keep explicit values
- `auto_calibrate_half_response="if_missing"`: fill missing `half_response` only
- `auto_calibrate_half_response=True`: always recalibrate before simulation

#### 3. Choose Trajectory

NVSim now uses a unified graph simulator. Provide a rooted state graph explicitly, then choose either a path-like or branching topology through graph construction helpers:

```python
import pandas as pd

from nvsim import path_graph, simulate
from nvsim.production import StateProductionProfile

graph = path_graph(["progenitor", "lineage_a"])
production = StateProductionProfile(
    pd.DataFrame({"g0": [1.0, 1.8]}, index=["progenitor", "lineage_a"])
)

result = simulate(
    grn,
    graph=graph,
    alpha_source_mode="state_anchor",
    production_profile=production,
    n_cells_per_state=50,
    root_time=2.0,
    state_time=2.0,
    dt=0.05,
    seed=7,
    master_regulators=["g0"],
    regulator_activity="spliced",
)
```

Use `path_graph([...])` for chain-like trajectories and `branching_graph(root, [...])` for one-to-many branching templates. For arbitrary rooted DAGs, pass a `StateGraph` or an edge table with `parent_state` / `child_state` columns.

#### 4. Choose Observed Noise

Observed-count generation supports two canonical capture models:

- `poisson_capture`: scale by `capture_rate`, then optionally sample Poisson counts
- `binomial_capture`: round latent counts, then apply molecule-level binomial capture

Recommended interpretation:

- `poisson_capture`: simple default for lightweight observation noise
- `binomial_capture`: closest observed-noise analogue to VeloSim technical noise

Main noise parameters:

- `capture_model`
- `capture_rate`
- `dropout_rate`
- `noise_seed`

Legacy aliases such as `scale_poisson` and `binomial` are compatibility-only names and should not be used in new code.

#### 5. Inspect Result

Simulation returns a plain dict with these main sections:

- `layers`: `true_unspliced`, `true_spliced`, `true_velocity`, `true_alpha`, plus observed `unspliced` / `spliced`
- `obs`: cell-level metadata such as pseudotime, branch labels, and sampling indices
- `var`: gene-level metadata such as `gene_role`, `gene_class`, `true_beta`, and `true_gamma`
- `uns`: configs and auxiliary metadata such as `true_grn`, `kinetic_params`, `simulation_config`, and `grn_calibration`

Use `to_anndata()` if you want an AnnData object for downstream analysis.

Recommended entry points:

```bash
python examples/tutorial.py
python examples/run_linear_continuous_program.py
python examples/run_bifurcation_state_anchor.py
python examples/plot_velocity_showcase.py
python examples/run_trrust_mouse_small.py
```

See [Alpha Source Modes](docs/alpha_source_modes.md) for formulas and examples.

### Migration Notes

The public API now uses canonical names consistently:

- `capture_model` replaces `noise_model`
- `poisson_capture` replaces legacy `scale_poisson`
- `binomial_capture` replaces legacy `binomial`
- `trunk_state` replaces `trunk_production_state`
- `branch_child_states` replaces `branch_production_states`
- `transition_schedule="linear"` replaces `interpolate_production=True`
- `K` replaces legacy `weight`
- `half_response` replaces legacy `threshold`

The old `linear` / `bifurcation` simulator special cases and their wrapper APIs have been removed. New code should always call `simulate(..., graph=...)` and express topology through `StateGraph`, `path_graph(...)`, or `branching_graph(...)`.

### Documentation

- [Alpha Source Modes](docs/alpha_source_modes.md)
- [Examples Guide](examples/README.md)

---

## 中文

### 项目概览

NVSim 是一个轻量的、GRN 感知的 RNA velocity 模拟器。

当前公开版本的定位，是一个基于确定性动力学的 benchmark scaffold，核心建模链路是：

```text
GRN -> alpha(t) -> unspliced/spliced ODE -> true velocity -> snapshot cells -> observed layers
```

它有意保持比 SERGIO、VeloSim、dyngen 或 scVelo 风格流程更小、更透明。目标不是做一个完整生物模拟器，而是提供一个便于检查 ground truth 的可读实现。

### 当前实现内容

- 显式 GRN 输入，包含 `regulator`、`target`、`sign`、`K`、`half_response`、`hill_coefficient`。
- 显式或自动推断的 master regulator。
- 面向 non-master gene 的 SERGIO 风格加性 Hill 生产率模型。
- 可配置的 regulator activity 调控活性代理：`spliced`（默认）、`unspliced` 或 `total = u + s`。
- 两种 master-regulator alpha source mode：`continuous_program` 保留 NVSim 原有连续时间程序，`state_anchor` 使用 SERGIO 风格 state/bin production anchor。
- 面向 unspliced / spliced RNA 的确定性 RNA velocity ODE。
- true layer 与 observed layer 分离。
- linear 和 trunk-to-two-branch bifurcation 两类模拟。
- 可选的 AnnData 导出和轻量可视化工具。

### 当前不做的内容

- SERGIO 的 CLE/SDE 随机模拟器。
- promoter switching。
- normal / MURK / branching gene classes。
- VeloSim 的 EVF-to-kinetics 映射。
- 完整 scVelo 风格 velocity embedding。

### 核心方程

对基因 `i`，当前模型使用：

```text
du_i/dt = alpha_i(t) - beta_i * u_i(t)
ds_i/dt = beta_i * u_i(t) - gamma_i * s_i(t)
v_i(t) = beta_i * u_i(t) - gamma_i * s_i(t)
```

其中：
- `alpha_i(t)` 是转录 / 生产率；
- `beta_i` 是剪接速率；
- `gamma_i` 是降解速率；
- `true_velocity` 对应 `ds/dt`；
- `true_velocity_u` 对应 `du/dt`。

### 源码地图

当前公开 API 采用扁平的 `nvsim/*.py` 布局，便于直接按模块阅读：

- `nvsim/grn.py`：GRN schema、master regulator 识别、graph level 和 half-response calibration helper。
- `nvsim/regulation.py`：SERGIO 风格 Hill activation/repression 与加性 target production。
- `nvsim/production.py`：master regulator forcing 定义，包括时间程序和 state/bin production profile。
- `nvsim/simulate.py`：`beta/gamma` 构造、`u0/s0` 初始状态校验、确定性 ODE 积分、snapshot sampling 和结果装配。
- `nvsim/noise.py`：canonical `poisson_capture` 与 `binomial_capture` 两类 observed-count 生成。
- `nvsim/output.py`：plain dict 输出和可选 AnnData 导出。
- `nvsim/plotting.py`：基于 scanpy/scVelo 的 velocity showcase workflow，以及 gene dynamics、phase portrait 等机制诊断图。
- `nvsim/sergio_io.py`：SERGIO `targets/regs` 输入解析。

### 安装

要求 Python 3.10 或以上。

```bash
pip install -e .
```

开发环境安装：

```bash
pip install -e .[dev]
```

### Public API Walkthrough

#### 1. Define GRN

先用 canonical 列定义 GRN：

- `regulator`
- `target`
- `sign`
- `K`
- `half_response`
- `hill_coefficient`

最小示例见上面的英文代码块，接口完全一致。

可读性上建议优先写 `sign="activation"` / `"repression"`；输入阶段也接受 `+` / `-`。

#### 2. Choose Alpha Source

当前 master regulator alpha 来源有两类：

- `continuous_program`：原始 NVSim 模式。master alpha 是连续时间函数，适合做干净、可控的 pseudotime benchmark。
- `state_anchor`：SERGIO 风格的 state/bin production anchor。每个 state/bin/cell type 有一套 master-regulator production vector，transition 可以用 `step`、`linear` 或 `sigmoid`。

`StateProductionProfile` 是用户提供的 simulation design input，可以手写、采样、复制自 SERGIO 风格 production table，或者由外部 cluster-level TF activity 粗略估计。

如果你在 bifurcation 中使用 `state_anchor`，推荐直接围绕这组 canonical 参数思考：

graph-based `state_anchor` 模式下，需要提供 graph 和与 graph state 对齐的 `production_profile`。每条 edge 都表示一次 parent-to-child 的 source production 过渡，`transition_schedule` 控制这条 segment 上的切换形状。分化轨迹默认更推荐 `transition_schedule="sigmoid"`，因为它比 hard step 更平滑。

Hill 调控中把哪一种 RNA 状态当作 regulator activity 也是显式可配的：

- 推荐默认：`regulator_activity="spliced"`
- 做 SERGIO 风格动态对照：`regulator_activity="unspliced"`
- 做敏感性分析：比较 `"spliced"`、`"unspliced"` 和 `"total"`

如果提供了 `StateProductionProfile`，还可以直接在模拟前自动补齐或重校准 `half_response`：

- `auto_calibrate_half_response=False`：保持显式输入
- `auto_calibrate_half_response="if_missing"`：只补缺失项
- `auto_calibrate_half_response=True`：总是在模拟前重校准

#### 3. Choose Trajectory

现在公开接口已经统一成 graph simulator。你需要显式给出 rooted state graph，再通过 helper 构建 path 或 branching 拓扑：

```python
import pandas as pd

from nvsim import path_graph, simulate
from nvsim.production import StateProductionProfile

graph = path_graph(["progenitor", "lineage_a"])
production = StateProductionProfile(
    pd.DataFrame({"g0": [1.0, 1.8]}, index=["progenitor", "lineage_a"])
)

result = simulate(
    grn,
    graph=graph,
    alpha_source_mode="state_anchor",
    production_profile=production,
    n_cells_per_state=50,
    root_time=2.0,
    state_time=2.0,
    dt=0.05,
    seed=7,
    master_regulators=["g0"],
    regulator_activity="spliced",
)
```

链式轨迹用 `path_graph([...])`；一对多分支可以用 `branching_graph(root, [...])`；更一般的 rooted DAG 可以直接传 `StateGraph` 或包含 `parent_state` / `child_state` 两列的 edge table。

#### 4. Choose Observed Noise

当前 observed noise 只有两种 canonical capture model：

- `poisson_capture`：先按 `capture_rate` 缩放，再可选做 Poisson 采样
- `binomial_capture`：先把 latent count 四舍五入，再做逐分子的 binomial capture

可以这样理解：

- `poisson_capture`：轻量默认噪声
- `binomial_capture`：在 observed noise 语义上最接近 VeloSim technical noise

最常调的噪声参数是：

- `capture_model`
- `capture_rate`
- `dropout_rate`
- `noise_seed`

`scale_poisson` 和 `binomial` 这些旧名字只保留兼容意义，新代码不建议再使用。

#### 5. Inspect Result

模拟结果是一个 plain dict，重点看四块：

- `layers`：`true_unspliced`、`true_spliced`、`true_velocity`、`true_alpha`，以及 observed `unspliced` / `spliced`
- `obs`：细胞级元数据，例如 pseudotime、branch、sampling index
- `var`：基因级元数据，例如 `gene_role`、`gene_class`、`true_beta`、`true_gamma`
- `uns`：配置和辅助元数据，例如 `true_grn`、`kinetic_params`、`simulation_config`、`grn_calibration`

如果你要接下游分析，可以用 `to_anndata()` 转成 AnnData。

建议从这些入口开始：

```bash
python examples/tutorial.py
python examples/run_linear_continuous_program.py
python examples/run_bifurcation_state_anchor.py
python examples/plot_velocity_showcase.py
python examples/run_trrust_mouse_small.py
```

详细公式和示例见 [Alpha Source Modes](docs/alpha_source_modes.md)。

### 迁移说明

当前公开 API 已统一使用 canonical 命名：

- `capture_model` 替代 `noise_model`
- `poisson_capture` 替代旧的 `scale_poisson`
- `binomial_capture` 替代旧的 `binomial`
- `trunk_state` 替代 `trunk_production_state`
- `branch_child_states` 替代 `branch_production_states`
- `transition_schedule="linear"` 替代 `interpolate_production=True`
- `K` 替代旧的 `weight`
- `half_response` 替代旧的 `threshold`

旧的 `linear` / `bifurcation` simulator 特例和对应 wrapper 已全部移除。新的调用方式统一为 `simulate(..., graph=...)`，拓扑通过 `StateGraph`、`path_graph(...)` 或 `branching_graph(...)` 表达。

### 相关文档

- [Alpha Source Modes](docs/alpha_source_modes.md)
- [Examples Guide](examples/README.md)
