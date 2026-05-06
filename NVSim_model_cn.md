# NVSim 模型说明：GRN 先验、RNA velocity 动力学与 SERGIO 的区别

本文档说明 NVSim 当前 v0.1 MVP 的模型细节、数据输出结构，以及它和 SERGIO 的关系。重点结论是：NVSim 可以读取 SERGIO/GNW 提供的 GRN 拓扑作为上游先验，但下游表达动力学、velocity 真值、噪声生成和 benchmark 输出都由 NVSim 自己定义。因此 `sergio_yeast400_*` 结果应理解为 **SERGIO-GRN-derived NVSim datasets**，不是 SERGIO simulation。

## 1. MVP 在 NVSim 中的含义

MVP 是 Minimum Viable Product，即“最小可用版本”。在 NVSim 中，它指当前版本只实现一条清晰、可验证、可导出真值的最小模拟链路：

```text
GRN -> alpha(t) -> unspliced/spliced ODE -> true velocity -> snapshot cells -> observed layers -> plots / AnnData
```

当前目标不是完整复刻 SERGIO、VeloSim、dyngen 或 scMultiSim，而是先提供一个透明、可控、工程化的 RNA velocity benchmark scaffold。

## 2. NVSim 的核心动力学模型

NVSim 的核心实现位于 `nvsim/simulate.py`。每个基因的状态包含 unspliced RNA 和 spliced RNA：

```text
u_i(t): gene i 的 unspliced RNA
s_i(t): gene i 的 spliced RNA
```

每个基因的动力学方程是：

```text
du_i/dt = alpha_i(t) - beta_i * u_i(t)
ds_i/dt = beta_i * u_i(t) - gamma_i * s_i(t)
true_velocity_i(t) = beta_i * u_i(t) - gamma_i * s_i(t)
```

其中：

- `alpha_i(t)` 是 transcription rate，由 master regulator program 或 GRN 调控函数决定。
- `beta_i` 是 splicing rate。
- `gamma_i` 是 degradation rate。
- `true_velocity_i(t)` 当前定义为成熟 RNA 的导数 `ds_i/dt`。

NVSim 使用固定时间步长的 RK4 integrator 积分。积分过程中会把 `u`、`s` 截断为非负值，防止数值误差产生负表达量。

这意味着 NVSim 当前是一个显式 ODE 模型，不是 molecule-level SSA，也不是 SERGIO 的 CLE/stochastic simulator。

## 3. GRN 在 NVSim 中怎么表示

GRN 的验证和标准化在 `nvsim/grn.py`。输入 edge table 必须包含：

```text
regulator, target, weight, sign
```

可选列包括：

```text
hill_coefficient, threshold
```

`sign` 会被标准化为两类：

```text
activation
repression
```

常见别名如 `+`、`1`、`act` 会转换为 `activation`；`-`、`-1`、`rep` 会转换为 `repression`。

一个重要设计是：NVSim 的 edge weight 必须是非负数，抑制边不通过负权重表示，而是通过 repression Hill response 表示。

## 4. GRN 如何决定 alpha(t)

调控函数实现在 `nvsim/regulation.py`。

对于 activation edge，NVSim 使用：

```text
H_act(x) = x^n / (h^n + x^n)
```

对于 repression edge，NVSim 使用：

```text
H_rep(x) = h^n / (h^n + x^n)
         = 1 - H_act(x)
```

其中：

- `x` 是 regulator 当前时刻的 spliced expression，即 `s_j(t)`。
- `h` 是 threshold。
- `n` 是 hill coefficient。

一个 target gene 的 transcription rate 是所有 incoming edges 的贡献之和，再加上 basal alpha：

```text
alpha_target(t) = basal_alpha_target + sum(edge_weight * Hill_response(s_regulator(t)))
```

对于 repression edge，贡献仍然是非负的：

```text
edge_weight * H_rep(s_regulator(t))
```

所以 regulator 越高，`H_rep` 越低，target 的 alpha 越低；但不会出现负 alpha。

## 5. Master regulator program

在 NVSim 中，没有 incoming edge 的基因被视为 master regulator。master regulator 的 `alpha(t)` 不由上游 GRN 决定，而是由显式时间程序决定。

这些程序在 `nvsim/programs.py` 中实现，当前包括：

```text
constant
linear_increase
linear_decrease
sigmoid_increase
sigmoid_decrease
```

master regulator program 接收 normalized pseudotime `t in [0, 1]`，输出对应时刻的 `alpha`。

target genes 则通过当前 regulator spliced state 和 GRN Hill response 动态计算 `alpha`。

这个设计让 NVSim 可以明确地控制 lineage-driving master regulators，再观察它们通过 GRN 传播到下游 target genes 的表达和 velocity。

## 6. Linear 和 bifurcation simulation

### 6.1 Linear

`simulate_linear()` 模拟一条单线轨迹。流程是：

```text
初始化 u0/s0/beta/gamma
在固定 time grid 上 RK4 积分
从 time grid 中抽样 snapshot cells
生成 true layers
生成 observed layers
返回 result dict 或 AnnData
```

输出 cell 的 `obs` 中包含：

```text
pseudotime
local_time
branch = linear
time_index
```

### 6.2 Bifurcation

`simulate_bifurcation()` 当前实现 trunk-to-two-branch MVP。流程是：

```text
1. 先模拟 trunk
2. 取 trunk 末端的 u/s 状态
3. 把 trunk terminal state 复制给 branch_0 和 branch_1 作为初始状态
4. branch_0 和 branch_1 分别独立积分
5. 抽样并拼接 cells：trunk -> branch_0 -> branch_1
```

branch-specific behavior 由 `branch_master_programs` 控制。如果没有传入 branch-specific programs，两个分支会从相同初始状态出发，并且可能保持非常相似。

输出 cell 的 `obs` 中包含：

```text
pseudotime
local_time
branch
segment
time_index
```

plain dictionary 输出还会包含：

```text
uns["segment_time_courses"]
uns["branch_inheritance"]
```

用于检查 branch 是否确实继承 trunk 末端状态。注意：这些内部 time-course 目前不会完整导出进 AnnData 的 `uns`。

## 7. 输出层和真值层

NVSim 的核心优势之一是区分 true layers 和 observed layers。输出矩阵方向统一为：

```text
cells x genes
```

主要 layers 包括：

```text
layers["unspliced"]       observed unspliced
layers["spliced"]         observed spliced
layers["true_unspliced"]  clean unspliced truth
layers["true_spliced"]    clean spliced truth
layers["true_velocity"]   ds/dt truth
layers["true_alpha"]      transcription-rate truth
```

`uns` 中包括：

```text
uns["true_grn"]
uns["kinetic_params"]
uns["simulation_config"]
```

其中 `kinetic_params` 当前包含 gene-level `beta` 和 `gamma`。

这种设计和很多 simulator 的结果不同：NVSim 不只给 count matrix，还直接保存了 velocity benchmark 需要的 ground truth。

## 8. 噪声模型

观测层由 `nvsim/noise.py` 生成。当前噪声模型很简单：

```text
true u/s
-> optional capture_rate scaling
-> optional Poisson sampling
-> optional dropout
-> observed u/s
```

参数包括：

```text
capture_rate
poisson_observed
dropout_rate
```

例如 `sergio_yeast400_*` 当前主 h5ad 使用：

```text
capture_rate = 0.6
dropout_rate = 0.01
poisson_observed = True
```

plot 脚本中还会生成 `observed_lownoise/` 视图，对应：

```text
capture_rate = 1.0
dropout_rate = 0.0
poisson_observed = False
```

这个低噪声输出主要用于可视化和 debug，不是现实 UMI 噪声模型。

## 9. NVSim 如何参考 SERGIO 的上游 GRN

NVSim 中与 SERGIO 相关的脚本主要是：

```text
examples/run_sergio_grn_bifurcation.py
examples/run_sergio_grn_multimaster.py
examples/plot_sergio_grn_bifurcation.py
examples/plot_sergio_grn_multimaster.py
```

它们使用的 SERGIO/GNW GRN 文件是：

```text
../SERGIO/GNW_sampled_GRNs/Yeast_400_net3.dot
```

也就是本地路径：

```text
/mnt/second19T/zhaozelin/simulator/SERGIO/GNW_sampled_GRNs/Yeast_400_net3.dot
```

这个 DOT 文件来自 SERGIO 仓库中的 GNW sampled GRNs。NVSim 读取其中的节点和边：

```text
"regulator" -> "target" [value="+"]
"regulator" -> "target" [value="-"]
```

然后转换为 NVSim 的 edge schema：

```text
regulator: DOT regulator node
target: DOT target node
sign: + -> activation, - -> repression
weight: NVSim 随机生成的非负 edge weight
threshold: NVSim 随机生成的 threshold
hill_coefficient: 2.0
```

当前转换函数是 `load_sergio_dot_grn()`。它保留了 SERGIO/GNW 的：

```text
基因集合
边方向
激活/抑制符号
GRN 拓扑结构
```

但不会保留 SERGIO 的表达动力学，也不会调用 SERGIO 的 simulation engine。

因此，上游 GRN 的意义是：NVSim 使用 SERGIO/GNW 的网络拓扑作为更真实/更大的调控图，而不是手工构造一个 toy GRN。

## 10. NVSim 下游如何和 SERGIO 区分

这是最关键的边界。

### 10.1 SERGIO-derived NVSim 不是 SERGIO simulation

`sergio_yeast400_3master` 和 `sergio_yeast400_multimaster` 这两个输出只是在上游读取了 SERGIO/GNW 的 Yeast-400 GRN。后续的：

```text
alpha(t) 计算
ODE 积分
unspliced/spliced dynamics
true_velocity 计算
snapshot cell sampling
observed count noise
AnnData 输出
quick-look plots
```

全部由 NVSim 完成。

所以这类结果更准确的名称是：

```text
SERGIO-GRN-derived NVSim datasets
```

不应称为 SERGIO datasets 或 SERGIO simulation results。

### 10.2 下游动力学机制不同

SERGIO 的重点是 stochastic gene expression simulation。它有自己的模拟框架、GRN 输入格式、technical noise 模块，以及 differentiation setting。

NVSim 当前则使用显式 ODE：

```text
du/dt = alpha - beta*u
ds/dt = beta*u - gamma*s
```

并且 `alpha` 是由 master regulator programs 和 GRN Hill response 计算出来的。

也就是说，即便上游 GRN 拓扑来自同一个 Yeast-400 DOT 文件，SERGIO 和 NVSim 的表达矩阵也不会等价，因为它们的下游动力学模型不同。

### 10.3 参数来源不同

在 SERGIO 中，GRN 文件和 SERGIO 输入文件会携带 SERGIO 自己需要的调控参数。

在当前 NVSim 的 SERGIO-GRN converter 中：

```text
edge weight: 由 NVSim 随机生成
threshold: 由 NVSim 随机生成
hill_coefficient: 固定为 2.0
beta/gamma: 由 NVSim 生成或用户传入
master programs: 由 NVSim example script 显式指定
```

因此，NVSim 参考的是 SERGIO/GNW 的网络结构，不是 SERGIO 的完整参数化模型。

### 10.4 输出真值不同

当前本地 SERGIO 输出文件如：

```text
simulator/SERGIO/results/sergio_cascade_final.h5ad
simulator/SERGIO/results/sergio_simulated.h5ad
```

主要包含：

```text
layers["spliced"]
layers["unspliced"]
obs["cell_type"]
```

而 NVSim 输出额外包含：

```text
layers["true_alpha"]
layers["true_unspliced"]
layers["true_spliced"]
layers["true_velocity"]
uns["true_grn"]
uns["kinetic_params"]
uns["simulation_config"]
```

这使 NVSim 更适合做方法开发和 benchmark，因为它把 ground truth 直接暴露出来。

### 10.5 噪声和稀疏性不同

当前本地 SERGIO 结果的观测层非常稀疏。例如已有结果中：

```text
sergio_cascade_final.h5ad:
  spliced zero fraction 约 94.1%
  unspliced zero fraction 约 98.7%
```

而当前 NVSim SERGIO-GRN results 中：

```text
sergio_yeast400_3master:
  spliced zero fraction 约 82.6%
  unspliced zero fraction 约 80.6%

sergio_yeast400_multimaster:
  spliced zero fraction 约 82.5%
  unspliced zero fraction 约 80.3%
```

这说明 NVSim 当前观测噪声比这些 SERGIO 输出更简单，也没有完全复现 SERGIO 的 dropout/library-size/noise 行为。

## 11. 当前四个输出结果说明

当前 `examples/outputs/` 下有四个主要结果。

### 11.1 linear_20gene

文件：

```text
examples/outputs/linear_20gene/mvp_linear.h5ad
```

规模：

```text
100 cells x 20 genes
```

用途：小型 linear sanity check。GRN 是手工构造的 20-gene toy GRN。

### 11.2 bifurcation_20gene_3master

文件：

```text
examples/outputs/bifurcation_20gene_3master/mvp_bifurcation.h5ad
```

规模：

```text
170 cells x 20 genes
trunk: 50
branch_0: 60
branch_1: 60
```

用途：小型 bifurcation sanity check。GRN 是手工构造的 20-gene toy GRN，含 3 个 master regulators。

### 11.3 sergio_yeast400_3master

文件：

```text
examples/outputs/sergio_yeast400_3master/sergio_yeast400_bifurcation.h5ad
```

规模：

```text
600 cells x 400 genes
trunk: 180
branch_0: 210
branch_1: 210
```

GRN：

```text
source_grn: SERGIO/GNW_sampled_GRNs/Yeast_400_net3.dot
n_genes: 400
n_edges: 1157
activation edges: 625
repression edges: 532
master regulators: 19
```

用于驱动 branch programs 的 master regulators：

```text
YIR018W
YDR146C
YNL216W
```

这个结果使用少数 high-outdegree master regulators 造成分支差异，因此分支信号相对低维。

### 11.4 sergio_yeast400_multimaster

文件：

```text
examples/outputs/sergio_yeast400_multimaster/sergio_yeast400_multimaster_bifurcation.h5ad
```

规模：

```text
740 cells x 400 genes
trunk: 220
branch_0: 260
branch_1: 260
```

GRN 与 `sergio_yeast400_3master` 相同，都是 Yeast-400 DOT。

用于驱动 branch programs 的 master regulators：

```text
YIR018W
YDR146C
YNL216W
YPL248C
YDR043C
YPL089C
YDL170W
YJL056C
YML113W
YPL177C
```

这个结果用 10 个 master regulators 驱动 branch-specific programs，因此真值层上的分支信号更高维，通常比 3-master 版本更丰富。

## 12. 两个 SERGIO-GRN 结果的经验差异

两个 SERGIO-GRN results 使用相同 GRN，但 branch-driving master 数量不同。

当前 PCA 空间中的 branch silhouette 约为：

```text
sergio_yeast400_3master:
  true_spliced PCA branch silhouette:    0.201
  observed_spliced PCA branch silhouette: 0.142

sergio_yeast400_multimaster:
  true_spliced PCA branch silhouette:    0.222
  observed_spliced PCA branch silhouette: 0.141
```

解释：

- multi-master 在 true layer 上确实产生了更强的分支结构。
- 但经过 capture/Poisson/dropout 后，observed layer 的分支分离优势被噪声削弱。
- 因此如果要看模型本身是否产生合理分支，应优先看 `plots/true/`；如果要看下游方法实际输入，应看 `plots/observed/`。

## 13. plot 文件夹说明

每个输出结果的 `plots/` 下通常有四类目录：

```text
true/
observed/
observed_lownoise/
diagnostics/
```

含义是：

```text
true/:
  使用 true layers 绘图，是主要科学验证视图。

observed/:
  使用 noisy observed layers 绘图，用于检查下游方法实际看到的数据。

observed_lownoise/:
  使用 capture_rate=1.0, dropout_rate=0.0, poisson_observed=False 的连续观测层。
  这是 debug/visualization 视图，不是现实 UMI 噪声。

diagnostics/:
  文本诊断，例如 selected genes、source GRN、edge 信息等。
```

## 14. 当前模型的优点和局限

### 优点

- 模型链路透明，公式简单，容易 debug。
- 输出 true alpha、true u/s、true velocity，适合 benchmark。
- 可以用真实/经典 GRN 拓扑作为先验，例如 SERGIO/GNW Yeast-400。
- 支持 linear 和 trunk-to-two-branch bifurcation。
- 真值层和观测层清楚分离。
- 测试覆盖了 GRN validation、velocity formula、non-negativity、seed reproducibility、bifurcation inheritance、AnnData export 等关键不变量。

### 局限

- 当前不是完整生物真实模拟器。
- 没有 promoter switching。
- 没有 molecule-level SSA。
- 没有 SERGIO CLE。
- 没有 VeloSim EVF-to-kinetics mapping。
- 没有正式 gene class 系统，如 branching genes、early/late response genes、housekeeping genes。
- 噪声模型较简单，没有 calibrated UMI realism。
- 当前 bifurcation 主要由人为指定的 master regulator programs 驱动。

## 15. 推荐表述

为了避免和 SERGIO 混淆，建议在论文、报告或文件命名中这样描述：

```text
NVSim uses the SERGIO/GNW Yeast-400 regulatory network as an upstream GRN prior, while generating RNA velocity dynamics, true kinetic layers, observed count layers, and branch programs through NVSim's own ODE-based simulation model.
```

中文可以写为：

```text
NVSim 使用 SERGIO/GNW Yeast-400 调控网络作为上游 GRN 拓扑先验，但下游的 alpha(t)、unspliced/spliced 动力学、true velocity、噪声观测层和分支程序均由 NVSim 自身的 ODE 模型生成。因此该数据是 SERGIO-GRN-derived NVSim 数据，而不是 SERGIO 模拟数据。
```
