"""Small configuration defaults used by NVSim internals."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GRNConfig:
    """GRN 边表缺失 Hill 参数时使用的默认值。"""

    default_hill_coefficient: float = 2.0
    default_threshold: float = 1.0
