"""master regulator 的 transcription-rate alpha 程序。

没有 incoming GRN edges 的基因被视为 master regulator。它们没有上游
regulator 可以计算 alpha，因此需要外部 alpha program 作为系统输入。
target/intermediate genes 的 alpha 仍然由 GRN 从当前 s(t) 计算。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

PROGRAM_KINDS = {
    "constant",
    "linear_increase",
    "linear_decrease",
    "sigmoid_increase",
    "sigmoid_decrease",
}


@dataclass(frozen=True)
class AlphaProgram:
    """一个 master regulator 的时间依赖 alpha 程序。

    ``t`` 使用归一化 pseudotime，范围 [0, 1]。不同 kind 表示常数、
    线性上升/下降、sigmoid 上升/下降。输出始终裁剪为非负。
    """

    kind: str = "constant"
    start: float = 1.0
    end: float = 1.0
    midpoint: float = 0.5
    steepness: float = 10.0

    def __post_init__(self) -> None:
        if self.kind not in PROGRAM_KINDS:
            raise ValueError(f"unknown alpha program kind {self.kind!r}")
        if self.start < 0 or self.end < 0:
            raise ValueError("alpha program start/end must be non-negative")
        if self.steepness <= 0:
            raise ValueError("alpha program steepness must be positive")

    def value(self, t: float) -> float:
        """在归一化 pseudotime ``t`` 上计算当前 master 的 alpha。"""

        t = float(np.clip(t, 0.0, 1.0))
        if self.kind == "constant":
            value = self.start
        elif self.kind == "linear_increase":
            value = self.start + (self.end - self.start) * t
        elif self.kind == "linear_decrease":
            value = self.end + (self.start - self.end) * (1.0 - t)
        elif self.kind == "sigmoid_increase":
            z = 1.0 / (1.0 + np.exp(-self.steepness * (t - self.midpoint)))
            value = self.start + (self.end - self.start) * z
        elif self.kind == "sigmoid_decrease":
            z = 1.0 / (1.0 + np.exp(-self.steepness * (t - self.midpoint)))
            value = self.end + (self.start - self.end) * (1.0 - z)
        else:
            raise ValueError(f"unknown alpha program kind {self.kind!r}")
        return float(max(value, 0.0))


def constant(value: float) -> AlphaProgram:
    return AlphaProgram(kind="constant", start=value, end=value)


def linear_increase(start: float, end: float) -> AlphaProgram:
    return AlphaProgram(kind="linear_increase", start=start, end=end)


def linear_decrease(start: float, end: float) -> AlphaProgram:
    return AlphaProgram(kind="linear_decrease", start=start, end=end)


def sigmoid_increase(start: float, end: float, midpoint: float = 0.5, steepness: float = 10.0) -> AlphaProgram:
    return AlphaProgram(kind="sigmoid_increase", start=start, end=end, midpoint=midpoint, steepness=steepness)


def sigmoid_decrease(start: float, end: float, midpoint: float = 0.5, steepness: float = 10.0) -> AlphaProgram:
    return AlphaProgram(kind="sigmoid_decrease", start=start, end=end, midpoint=midpoint, steepness=steepness)


def coerce_programs(programs: Mapping[str, AlphaProgram | float] | None) -> dict[str, AlphaProgram]:
    """标准化用户输入的 master programs。

    用户可以传入 AlphaProgram，也可以直接传入数字；数字会被解释成
    constant alpha program。
    """

    if programs is None:
        return {}
    coerced: dict[str, AlphaProgram] = {}
    for gene, program in programs.items():
        if isinstance(program, AlphaProgram):
            coerced[str(gene)] = program
        else:
            coerced[str(gene)] = constant(float(program))
    return coerced


def evaluate_programs(
    programs: Mapping[str, AlphaProgram | float],
    genes: list[str] | tuple[str, ...],
    t: float,
    default: float = 0.5,
) -> pd.Series:
    """按照固定 gene 顺序批量计算 master programs，返回 pandas Series。"""

    coerced = coerce_programs(programs)
    values = []
    for gene in genes:
        program = coerced.get(str(gene), constant(default))
        values.append(program.value(t))
    return pd.Series(values, index=pd.Index([str(g) for g in genes], name="gene"), dtype=float)
