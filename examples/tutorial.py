"""当前 NVSim public simulation API 的规范教程。

这个文件建议作为“复制示例调用”的首选位置：
- 如果 public API（公开接口）变化，应该在同一次修改里同步更新本教程。
- 本教程只使用 canonical API（规范接口），尽量避免旧参数名。
- 中文注释重点解释每个参数的含义、可选值和使用场景。

核心建模链：
    GRN -> alpha(t) -> unspliced u(t) -> spliced s(t) -> true velocity

其中：
- GRN：gene regulatory network，基因调控网络。
- alpha(t)：transcription / production rate，转录/生成速率。
- unspliced u(t)：未剪接 RNA。
- spliced s(t)：已剪接 RNA。
- true_velocity：真实 RNA velocity，当前定义为 ds/dt = beta * u - gamma * s。
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

# 让这个脚本可以直接从 examples/ 目录运行，而不需要先 pip install -e .
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nvsim.grn import GRN
from nvsim.output import to_anndata
from nvsim.production import StateProductionProfile, linear_increase, sigmoid_decrease
from nvsim.simulate import simulate


def build_tutorial_grn() -> GRN:
    """创建一个最小 GRN 示例，使用当前推荐的 canonical edge schema。

    当前推荐的 GRN 边表字段：
    - regulator：上游调控基因。
    - target：被调控的目标基因。
    - sign：调控方向，可选 "activation" 或 "repression"。
        activation：激活调控，上游越高，目标 alpha 越高。
        repression：抑制调控，上游越高，目标 alpha 越低。
    - K：调控强度，对应 SERGIO-style Hill contribution 的幅度。
    - half_response：半响应值，也就是 Hill 函数达到半饱和时的 regulator 表达量。
      旧名 threshold 不再建议用于新代码。
    - hill_coefficient：Hill 系数，控制响应曲线陡峭程度。

    注意：
    - weight / threshold 是旧接口别名，新教程中不要再使用。
    - master_regulators 是主调控基因，通常是 GRN 中没有上游输入、由外部 alpha 程序驱动的基因。
    """

    edges = pd.DataFrame(
        {
            # g0, g1, g2 分别调控 g3, g4, g5
            "regulator": ["g0", "g1", "g2"],
            "target": ["g3", "g4", "g5"],

            # sign 可选：
            # - "activation"：激活
            # - "repression"：抑制
            "sign": ["activation", "activation", "repression"],

            # K：调控边强度；数值越大，该边对 target alpha 的贡献越大。
            # 当前实现要求非负，激活/抑制方向由 sign 控制，不靠 K 的正负控制。
            "K": [0.8, 0.6, 0.5],

            # half_response：Hill 函数的半响应位置。
            # regulator 表达量接近该值时，Hill 响应大约处于中间水平。
            "half_response": [0.5, 0.6, 0.5],

            # hill_coefficient：Hill 曲线陡峭程度。
            # 值越大，响应越接近开关；值越小，响应越平滑。
            "hill_coefficient": [2.0, 2.0, 2.0],
        }
    )

    # genes 决定模拟矩阵的基因顺序；所有 layer 都会按这个顺序排列。
    genes = [f"g{i}" for i in range(6)]

    # master regulators：主调控基因。
    # 在当前教程中，g0/g1/g2 不由 GRN 入边决定 alpha，
    # 而是由 master_programs 或 StateProductionProfile 外部指定 alpha。
    masters = ["g0", "g1", "g2"]

    return GRN.from_dataframe(edges, genes=genes, master_regulators=masters)


def build_tutorial_profile() -> StateProductionProfile:
    """构建 state-anchor 模式使用的 production profile。

    StateProductionProfile 是“状态锚点生产率表”：
    - 行：离散状态，例如 root、branch_A、branch_B。
    - 列：master regulator，例如 g0、g1、g2。
    - 值：该状态下对应 master regulator 的 alpha / production rate。

    使用场景：
    - alpha_source_mode="state_anchor" 时使用。
    - 很适合模拟 SERGIO-style 的状态/分支锚点：
      例如 root 状态有一组 master alpha，两个 branch 终点有不同 master alpha。

    与 continuous_program 的区别：
    - continuous_program：每个 master gene 自己有一个连续时间函数 alpha_m(t)。
    - state_anchor：每个状态有一组固定 alpha，branch 过程可以在 parent/child state 之间插值。
    """

    return StateProductionProfile(
        pd.DataFrame(
            {
                # 每一列是一个 master regulator 的 alpha。
                # 每一行是一个状态锚点。
                "g0": [0.4, 1.2, 0.2],
                "g1": [0.8, 0.4, 1.1],
                "g2": [0.7, 1.0, 0.3],
            },
            # root：trunk 起点/根状态。
            # branch_A / branch_B：两个分支的目标状态。
            index=["root", "branch_A", "branch_B"],
        )
    )


def linear_parameters() -> dict[str, object]:
    """线性轨迹 tutorial 的推荐参数块。

    simulate(..., simulator="linear") 用于生成一条线性轨迹：
        root / early cells -> later cells

    这里使用 alpha_source_mode="continuous_program"，
    即每个 master regulator 的 alpha 由连续时间程序给出。
    """

    return {
        # n_cells：从连续 ODE time course 中抽取多少个 snapshot cells。
        # 注意：如果 n_cells 大于可用时间点数量，而不允许 replacement，会报错。
        # 增大 n_cells：细胞更多，轨迹采样更密。
        "n_cells": 60,

        # time_end：ODE 积分的终止时间。
        # 时间越长，u/s 状态有更充分时间响应 alpha 的变化。
        "time_end": 3.0,

        # dt：ODE 数值积分步长。
        # 越小越精确，但计算更慢；当前 tutorial 用 0.05 作为较稳妥示例。
        "dt": 0.05,

        # alpha_source_mode：master regulator 的 alpha 来源。
        # 当前推荐可选：
        # - "continuous_program"：连续时间程序，使用 master_programs。
        # - "state_anchor"：状态锚点表，使用 production_profile。
        #
        # 线性 tutorial 使用 continuous_program。
        "alpha_source_mode": "continuous_program",

        # master_programs：只在 alpha_source_mode="continuous_program" 下使用。
        # 用来指定 master regulators 的 alpha_m(t)。
        #
        # 可用 helper：
        # - constant(value)：常数 alpha。
        # - linear_increase(start, end)：线性上升。
        # - linear_decrease(start, end)：线性下降。
        # - sigmoid_increase(start, end, midpoint, steepness)：sigmoid 上升。
        # - sigmoid_decrease(start, end, midpoint, steepness)：sigmoid 下降。
        #
        # 也可以直接传 float，例如 "g1": 0.8 会被解释为 constant(0.8)。
        "master_programs": {
            # g0 的 alpha 从 0.2 线性升到 1.0。
            "g0": linear_increase(0.2, 1.0),

            # g1 的 alpha 固定为 0.8。
            "g1": 0.8,

            # g2 的 alpha 从 1.0 sigmoid 下降到 0.3。
            "g2": sigmoid_decrease(1.0, 0.3),
        },

        # regulator_activity：GRN 计算 alpha 时，regulator 的“活性”用什么表达量表示。
        # 当前可选：
        # - "spliced"：使用 spliced s(t)，默认/推荐，更接近成熟 mRNA 调控假设。
        # - "unspliced"：使用 unspliced u(t)，更偏 nascent RNA 活性。
        # - "total"：使用 u(t) + s(t)。
        #
        # 当前推荐先用 "spliced"，因为解释最稳定。
        "regulator_activity": "spliced",

        # capture_model：观测层噪声的捕获模型。
        # 当前 canonical choices（规范取值）：
        # - "poisson_capture"：
        #     先按 capture_rate 缩放 true counts，再做 Poisson 采样。
        #     适合模拟 UMI count 的随机采样。
        # - "binomial_capture"：
        #     VeloSim-style molecule capture：
        #     observed ~ Binomial(round(true), capture_rate)。
        #     适合复现 VeloSim 类的分子捕获/dropout 效果。
        #
        # 旧别名如 "scale_poisson" / "binomial" 不建议在新教程中使用。
        "capture_model": "poisson_capture",

        # capture_rate：捕获率，范围通常在 [0, 1]。
        # - 越低：observed counts 更稀疏，zero 更多。
        # - 越高：observed 更接近 true layer。
        # capture_rate=1.0 可作为高捕获/近似 clean observed control。
        "capture_rate": 0.5,

        # poisson_observed：是否把缩放后的连续值通过 Poisson 采样变成整数 count。
        # - True：更像真实 observed count。
        # - False：保留连续值，适合 debug 或低噪声可视化。
        #
        # 注意：
        # - 主要影响 capture_model="poisson_capture"。
        # - 对 "binomial_capture" 来说，binomial 本身已经产生离散观测。
        "poisson_observed": True,

        # dropout_rate：额外的全局 dropout 概率。
        # 当前是 uniform/global dropout，也就是所有 cell-gene entry 用同一个概率随机置零。
        # - 0.0：不额外 dropout。
        # - 0.02：约 2% entry 额外置零。
        #
        # 注意：
        # 当前它不是 mean-dependent dropout（表达量依赖 dropout）。
        # 如果使用 binomial_capture，低 capture_rate 本身已经会制造很多 zero，
        # 再加 dropout_rate 可能导致“二次掉落”。
        "dropout_rate": 0.02,

        # seed：随机种子。
        # 影响 beta/gamma 随机初始化、初始 u0/s0、snapshot sampling 和 observed noise 等。
        # 固定 seed 可以复现实验结果。
        "seed": 7,
    }


def bifurcation_parameters() -> dict[str, object]:
    """state-anchor bifurcation tutorial 的推荐参数块。

    simulate(..., simulator="bifurcation") 用于生成 trunk-to-two-branch 轨迹：
        trunk -> branch_0
              -> branch_1

    这里使用 alpha_source_mode="state_anchor"，
    即 master regulator 的 alpha 来自 StateProductionProfile 中的状态锚点。
    """

    return {
        # n_trunk_cells：trunk 段抽取多少个 snapshot cells。
        # trunk 是分支前的共同祖先轨迹。
        "n_trunk_cells": 30,

        # n_branch_cells：每个 branch 抽取多少个 snapshot cells。
        # 推荐用 dict 明确指定 branch_0 / branch_1，避免歧义。
        # 当前 tutorial 是 branch_0 和 branch_1 各 40 个细胞。
        "n_branch_cells": {"branch_0": 40, "branch_1": 40},

        # trunk_time：trunk 段 ODE 积分时长。
        # 决定共同祖先状态演化多久后进入分支。
        "trunk_time": 1.8,

        # branch_time：每个 branch 段 ODE 积分时长。
        # 决定分支状态从 trunk terminal state 向 branch-specific state 演化多久。
        "branch_time": 2.2,

        # dt：ODE 数值积分步长。
        # 和 linear 一样，越小越精确，越大越快但可能更粗糙。
        "dt": 0.05,

        # alpha_source_mode：master alpha 来源。
        # 对 bifurcation 的 state-anchor 教程，应使用 "state_anchor"。
        #
        # 可选：
        # - "continuous_program"：连续 alpha 程序。
        # - "state_anchor"：状态锚点 alpha 表。
        "alpha_source_mode": "state_anchor",

        # production_profile：状态锚点生产率表。
        # 必须包含 master regulators 对应列，例如 g0/g1/g2。
        # 必须包含下面 trunk_state 和 branch_child_states 中引用的状态行。
        "production_profile": build_tutorial_profile(),

        # trunk_state：trunk 使用的 parent/root 状态。
        # 这里对应 production_profile 的 index="root"。
        "trunk_state": "root",

        # branch_child_states：每个 branch 的 child/target 状态。
        # branch_0 会从 trunk_state 过渡到 branch_A。
        # branch_1 会从 trunk_state 过渡到 branch_B。
        #
        # 这比旧接口 trunk_production_state / branch_production_states 更清楚，
        # 新代码建议使用这个 canonical 参数。
        "branch_child_states": {"branch_0": "branch_A", "branch_1": "branch_B"},

        # transition_schedule：从 trunk_state alpha 过渡到 branch child state alpha 的方式。
        # 当前可选：
        # - "step"：阶跃切换；到 midpoint 前是 parent，之后是 child。
        # - "linear"：线性插值；从 parent 平滑线性变到 child。
        # - "sigmoid"：sigmoid 平滑切换；更像软开关。
        #
        # 推荐：
        # - 想模拟突然命运决定：用 "step"。
        # - 想模拟均匀连续过渡：用 "linear"。
        # - 想模拟较自然的开关式连续过渡：用 "sigmoid"。
        "transition_schedule": "sigmoid",

        # transition_midpoint：过渡中心点，范围通常为 [0, 1]。
        # - 0.5 表示在 branch 段中间完成主要切换。
        # - 对 "step" 是阶跃位置。
        # - 对 "sigmoid" 是 sigmoid 中心。
        # - 对 "linear" 基本不敏感。
        "transition_midpoint": 0.5,

        # transition_steepness：sigmoid 过渡陡峭程度。
        # - 数值越大，切换越突然。
        # - 数值越小，切换越平缓。
        # 主要用于 transition_schedule="sigmoid"。
        "transition_steepness": 10.0,

        # regulator_activity：计算 target alpha 时，用哪种 RNA 层表示 regulator 活性。
        # 可选：
        # - "spliced"
        # - "unspliced"
        # - "total"
        #
        # bifurcation 中仍推荐先用 "spliced"。
        "regulator_activity": "spliced",

        # capture_model：观测层 capture 模型。
        # 这里使用 "binomial_capture"，等价于 VeloSim-style：
        # observed ~ Binomial(round(true), capture_rate)
        #
        # 如果你想和 VeloSim 噪声更接近，优先使用这个模式。
        "capture_model": "binomial_capture",

        # capture_rate：分子捕获率。
        # 对 binomial_capture 而言：
        # - 0.3 表示每个 latent molecule 约有 30% 概率被观测到。
        # - capture_rate 越低，zero/dropout 越强。
        "capture_rate": 0.3,

        # dropout_rate：额外全局 dropout。
        # 这里设为 0.0，因为 binomial_capture 已经会自然产生很多 zero。
        # 如果再加全局 dropout，可能会比 VeloSim 更稀疏。
        "dropout_rate": 0.0,

        # seed：随机种子。
        # 固定后可复现 trunk/branch snapshot sampling 和 observed layer。
        "seed": 11,
    }


def run_linear_tutorial() -> dict:
    """运行线性轨迹模拟。

    返回值是 plain dict，而不是强制 AnnData。
    这样即使没有安装 anndata，也可以运行模拟和检查 layers/obs/var/uns。
    """

    return simulate(build_tutorial_grn(), simulator="linear", **linear_parameters())


def run_bifurcation_tutorial() -> dict:
    """运行 trunk-to-two-branch 分支轨迹模拟。"""

    return simulate(build_tutorial_grn(), simulator="bifurcation", **bifurcation_parameters())


def _write_if_possible(result: dict, path: Path) -> None:
    """如果安装了 anndata，则把 result dict 写成 h5ad。

    h5ad 是 AnnData 的常用文件格式，便于后续接 Scanpy/scVelo。
    如果用户环境没有 anndata，本函数只跳过导出，不影响模拟本身。
    """

    try:
        adata = to_anndata(result)
    except ImportError:
        print(f"anndata is not installed; skipped {path.name}")
        return
    adata.write_h5ad(path)


def _print_summary(name: str, result: dict) -> None:
    """打印一个简短摘要，确认关键接口参数是否按预期进入 simulation_config。"""

    config = result["uns"]["simulation_config"]

    print(f"[{name}]")
    print(f"  n_cells={result['obs'].shape[0]}")
    print(f"  n_genes={result['var'].shape[0]}")

    # alpha_source_mode：
    # - continuous_program：master alpha 来自连续时间程序。
    # - state_anchor：master alpha 来自状态锚点表。
    print(f"  alpha_source_mode={config['alpha_source_mode']}")

    # capture_model：
    # - poisson_capture
    # - binomial_capture
    print(f"  capture_model={config['capture_model']}")

    # regulator_activity：
    # - spliced
    # - unspliced
    # - total
    print(f"  regulator_activity={config['regulator_activity']}")

    # bifurcation 专属字段。
    if "trunk_state" in config:
        print(f"  trunk_state={config['trunk_state']}")
        print(f"  branch_child_states={config['branch_child_states']}")
        print(f"  transition_schedule={config['transition_schedule']}")


def main() -> None:
    """运行两个 tutorial 示例并保存输出。"""

    out_dir = Path(__file__).with_name("outputs") / "tutorial"
    out_dir.mkdir(parents=True, exist_ok=True)

    linear_result = run_linear_tutorial()
    bifurcation_result = run_bifurcation_tutorial()

    # 如果安装了 anndata，会写出：
    # - tutorial_linear.h5ad
    # - tutorial_bifurcation.h5ad
    #
    # 否则只打印提示，不影响前面的模拟。
    _write_if_possible(linear_result, out_dir / "tutorial_linear.h5ad")
    _write_if_possible(bifurcation_result, out_dir / "tutorial_bifurcation.h5ad")

    _print_summary("linear", linear_result)
    _print_summary("bifurcation", bifurcation_result)
    print(f"outputs={out_dir}")


if __name__ == "__main__":
    main()
