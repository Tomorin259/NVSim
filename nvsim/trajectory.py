"""NVSim MVP 的轨迹元数据构造工具。

这里生成的是 cell-level metadata，例如 pseudotime、branch、local_time。
真正的 ODE 积分在 ``simulate.py`` 中完成。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_count(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def make_linear_trajectory(
    n_cells: int,
    branch: str = "root",
    start: float = 0.0,
    end: float = 1.0,
) -> pd.DataFrame:
    """生成线性轨迹的伪时间 metadata。

    返回的每一行对应一个 cell。``pseudotime`` 是全局坐标；
    ``local_time`` 和 ``local_pseudotime`` 是当前 segment 内部坐标。
    在线性轨迹里二者通常等价。
    """

    n_cells = _validate_count("n_cells", n_cells)
    local = np.linspace(0.0, 1.0, n_cells)
    pseudotime = np.linspace(start, end, n_cells)
    return pd.DataFrame(
        {
            "cell_id": [f"{branch}_{idx}" for idx in range(n_cells)],
            "pseudotime": pseudotime,
            "branch": branch,
            "parent_branch": pd.NA,
            "local_pseudotime": local,
            "local_time": local,
            "is_trunk": branch == "root",
        }
    )


def make_bifurcation_trajectory(
    n_trunk_cells: int,
    n_branch_cells: int | dict[str, int],
    branches: tuple[str, ...] = ("branch_0", "branch_1"),
    trunk_branch: str = "trunk",
) -> pd.DataFrame:
    """生成 trunk + 多个 branch 的分叉轨迹 metadata。

    行顺序代表采样顺序：先 trunk，再按 ``branches`` 顺序拼接各 branch。
    科学逻辑是先积分 trunk，在分叉点复制 terminal u/s 状态，然后每条
    branch 独立积分。``pseudotime`` 是全局时间；``local_time`` 是每个
    segment 内部从 0 开始的局部时间。
    """

    n_trunk_cells = _validate_count("n_trunk_cells", n_trunk_cells)
    if not branches:
        raise ValueError("at least one branch is required")

    trunk = make_linear_trajectory(n_trunk_cells, branch=trunk_branch, start=0.0, end=1.0)
    trunk["parent_branch"] = pd.NA
    trunk["is_trunk"] = True
    trunk["branch_point"] = False
    trunk["segment_order"] = 0
    trunk.loc[trunk.index[-1], "branch_point"] = True

    branch_frames = []
    for order, branch in enumerate(branches, start=1):
        count = n_branch_cells[branch] if isinstance(n_branch_cells, dict) else n_branch_cells
        count = _validate_count(f"n_branch_cells[{branch}]", count)
        frame = make_linear_trajectory(count, branch=branch, start=1.0, end=2.0)
        frame["parent_branch"] = trunk_branch
        frame["is_trunk"] = False
        frame["branch_point"] = False
        frame["segment_order"] = order
        branch_frames.append(frame)

    result = pd.concat([trunk, *branch_frames], ignore_index=True)
    result["cell_id"] = [f"cell_{idx}" for idx in range(len(result))]
    result["sample_order"] = np.arange(len(result))
    return result
