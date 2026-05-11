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
- Linear and trunk-to-two-branch bifurcation simulation.
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

- `nvsim/grn.py`: GRN schema, master-regulator detection, graph levels, and threshold calibration helpers.
- `nvsim/regulation.py`: SERGIO-style Hill activation/repression and additive target production.
- `nvsim/production.py`: master-regulator forcing definitions, including time-dependent alpha programs and state/bin-wise production profiles.
- `nvsim/simulate.py`: beta/gamma setup, initial `u0/s0` validation, deterministic ODE integration, snapshot sampling, and result assembly.
- `nvsim/noise.py`: observed-count generation for `scale_poisson` and `binomial_capture`.
- `nvsim/output.py`: plain dict output and optional AnnData export.
- `nvsim/plotting.py`: quick-look PCA/UMAP, phase portraits, dynamics, and gallery plots.
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

### Quick Start

Run the small linear example:

```bash
python examples/run_mvp_linear.py
python examples/plot_linear.py
```

Run the bifurcation example:

```bash
python examples/run_mvp_bifurcation.py
python examples/plot_bifurcation.py
```

Minimal Python usage:

```python
import pandas as pd

from nvsim.grn import GRN
from nvsim.production import StateProductionProfile
from nvsim.simulate import simulate_linear

edges = pd.DataFrame(
    {
        "regulator": ["g0"],
        "target": ["g1"],
        "K": [0.8],
        "sign": ["activation"],
        "half_response": [0.5],
        "hill_coefficient": [2.0],
    }
)

grn = GRN.from_dataframe(edges, genes=["g0", "g1"])
production = StateProductionProfile(
    pd.DataFrame({"g0": [1.0]}, index=["state_0"])
)
result = simulate_linear(
    grn,
    n_cells=50,
    time_end=2.0,
    dt=0.05,
    seed=7,
    master_regulators=["g0"],
    production_profile=production,
    production_state="state_0",
    regulator_activity="spliced",
)

print(result["layers"]["true_spliced"].shape)
print(result["var"][["gene_role", "gene_class"]].head())
```

### Current Output Modes

Observed-count generation currently supports:

- `scale_poisson`
- `binomial_capture`

Metadata such as `grn_calibration` and `noise_config` is stored in the plain result dict and carried into AnnData output.

### Master-Regulator Alpha Source Modes

NVSim supports two master-regulator alpha source modes:

- `continuous_program`: the original NVSim mode. Master regulator alpha is a
  time function, `alpha_m(t) = f_m(t)`. This is useful for clean, controlled
  pseudotime velocity benchmarks.
- `state_anchor`: a SERGIO-inspired mode. Each state/bin/cell type has a
  master-regulator production vector, and transitions can interpolate from a
  parent state to a child state with `step`, `linear`, or `sigmoid` schedules.
  `sigmoid` is recommended for differentiation-like trajectories because it
  avoids hard production-rate jumps.

See [Alpha Source Modes](docs/alpha_source_modes.md) for formulas and examples.

NVSim uses SERGIO-style additive Hill regulation, but the default
`regulator_activity="spliced"` is not the strict SERGIO-compatible dynamic
regulator proxy.

Regulatory contributions can choose which RNA state is treated as regulator activity:

- Recommended default: `regulator_activity="spliced"`; uses current `s(t)` as a mature-mRNA / downstream biological proxy.
- For SERGIO-compatible dynamic comparison runs: `regulator_activity="unspliced"`; uses current `u(t)` as a closer expression-concentration proxy.
- For sensitivity analysis: compare `regulator_activity="spliced"`, `"unspliced"`, and `"total"`; `total` uses `u(t) + s(t)` as a total-RNA proxy.

Half-response calibration is no longer limited to a separate preprocessing step.
If a `StateProductionProfile` is available, `simulate_linear()` and
`simulate_bifurcation()` can now run:

- `auto_calibrate_half_response=False` (default): keep the old explicit behavior;
- `auto_calibrate_half_response="if_missing"`: fill missing `half_response` only;
- `auto_calibrate_half_response=True`: always recalibrate from the state/bin
  production matrix before simulation.

### Documentation

- [Current Status](CURRENT_STATUS.md)
- [Validation Report](VALIDATION_REPORT.md)
- [Alpha Source Modes](docs/alpha_source_modes.md)
- [Chinese Model Notes](NVSim_model_cn.md)
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

- `nvsim/grn.py`：GRN schema、master regulator 识别、graph level 和 threshold calibration helper。
- `nvsim/regulation.py`：SERGIO 风格 Hill activation/repression 与加性 target production。
- `nvsim/production.py`：master regulator forcing 定义，包括时间程序和 state/bin production profile。
- `nvsim/simulate.py`：`beta/gamma` 构造、`u0/s0` 初始状态校验、确定性 ODE 积分、snapshot sampling 和结果装配。
- `nvsim/noise.py`：`scale_poisson` 与 `binomial_capture` 两类 observed-count 生成。
- `nvsim/output.py`：plain dict 输出和可选 AnnData 导出。
- `nvsim/plotting.py`：PCA/UMAP、phase portrait、dynamics 和缩略图库绘图。
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

### 快速开始

运行小型 linear example：

```bash
python examples/run_mvp_linear.py
python examples/plot_linear.py
```

运行 bifurcation example：

```bash
python examples/run_mvp_bifurcation.py
python examples/plot_bifurcation.py
```

最小 Python 调用示例见上面的英文部分，接口相同。

### 当前支持的 observed-count 模式

目前支持两种 observed count generation：

- `scale_poisson`
- `binomial_capture`

同时，`grn_calibration` 和 `noise_config` 会保存在 result dict 中，并在导出 AnnData 时保留下来。

### Master-Regulator Alpha Source Modes

NVSim 现在支持两种 master regulator alpha 来源：

- `continuous_program`：原有 NVSim 模式。master regulator alpha 是连续时间函数，即 `alpha_m(t) = f_m(t)`，适合干净、机制可控的 pseudotime velocity benchmark。
- `state_anchor`：借鉴 SERGIO 的 state/bin production 设计。每个 state/bin/cell type 有一套 master-regulator production vector；transition 时可以用 `step`、`linear` 或 `sigmoid` 从 parent state 平滑过渡到 child state。分化轨迹默认推荐 `sigmoid`，避免 hard switching 造成表达或 embedding 断裂。

详细公式和示例见 [Alpha Source Modes](docs/alpha_source_modes.md)。

当前仍然是 SERGIO 风格的加性 Hill regulation，但默认的
`regulator_activity="spliced"` 并不是严格的 SERGIO-compatible dynamic
regulator proxy。

Hill 调控里使用哪一种 RNA 状态作为 regulator activity 现在也是显式可配的：

- 推荐默认：`regulator_activity="spliced"`，用当前 `s(t)` 作为 mature mRNA / downstream proxy；
- 做 SERGIO-compatible dynamic 对照时：`regulator_activity="unspliced"`，用当前 `u(t)`，更接近 SERGIO 风格的表达浓度代理；
- 做 sensitivity analysis 时：比较 `regulator_activity="spliced"`、`"unspliced"` 和 `"total"`；其中 `total` 用 `u(t) + s(t)` 作为 total RNA proxy。

half-response calibration 也不再只能作为单独预处理步骤使用。
如果提供了 `StateProductionProfile`，现在可以在
`simulate_linear()` / `simulate_bifurcation()` 中直接使用：

- `auto_calibrate_half_response=False`：保持原来的显式预处理行为；
- `auto_calibrate_half_response="if_missing"`：只在缺失 `half_response` 时自动补齐；
- `auto_calibrate_half_response=True`：在模拟前根据 state/bin production matrix 主动重校准。

### 相关文档

- [Current Status](CURRENT_STATUS.md)
- [Validation Report](VALIDATION_REPORT.md)
- [Alpha Source Modes](docs/alpha_source_modes.md)
- [Chinese Model Notes](NVSim_model_cn.md)
- [Examples Guide](examples/README.md)
