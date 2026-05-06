"""NVSim MVP 使用的轻量配置对象。

这些 dataclass 主要保存默认参数，避免把 magic numbers 分散在各模块。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GRNConfig:
    """GRN 边表缺失 Hill 参数时使用的默认值。"""

    default_hill_coefficient: float = 2.0
    default_threshold: float = 1.0


@dataclass(frozen=True)
class TrajectoryConfig:
    """简单轨迹 metadata 生成时使用的默认配置。"""

    n_trunk_cells: int = 50
    n_branch_cells: int = 50
    branch_names: tuple[str, ...] = ("branch_1", "branch_2")


@dataclass(frozen=True)
class SimulationConfig:
    """顶层模拟配置，目前主要用于保存随机种子、步长和子配置。"""

    seed: int = 0
    dt: float = 0.01
    grn: GRNConfig = GRNConfig()
    trajectory: TrajectoryConfig = TrajectoryConfig()

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be positive")
