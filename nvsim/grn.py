"""NVSim 的 GRN（gene regulatory network）表示与校验。

本模块只负责“网络是什么”，不负责真正计算动力学。
一张 GRN 表由多条 regulator -> target 边组成，每条边必须包含：
regulator、target、weight、sign。可选列 hill_coefficient 和 half_response
控制 Hill 调控响应的形状。

重要约定：weight 永远是非负数；activation/repression 只决定使用
H_act(x) 还是 H_rep(x)。抑制边不再额外乘负号，这样 alpha 始终可以
被解释为非负的 transcription-rate contribution。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import GRNConfig

REQUIRED_COLUMNS = ("regulator", "target", "weight", "sign")
OPTIONAL_COLUMNS = ("hill_coefficient", "half_response", "threshold")
RETURN_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
VALID_SIGNS = {"activation", "repression"}
_SIGN_ALIASES = {
    "activation": "activation",
    "activating": "activation",
    "act": "activation",
    "+": "activation",
    "+1": "activation",
    "1": "activation",
    "repression": "repression",
    "repressive": "repression",
    "rep": "repression",
    "-": "repression",
    "-1": "repression",
}


def normalize_sign(value: object) -> str:
    """把用户常见的边符号统一成 activation 或 repression。

例如 ``+``、``act`` 会被标准化成 ``activation``；``-``、``rep``
会被标准化成 ``repression``。这样后续计算不需要处理多种写法。
"""

    key = str(value).strip().lower()
    if key not in _SIGN_ALIASES:
        raise ValueError(f"invalid edge sign {value!r}; expected activation or repression")
    return _SIGN_ALIASES[key]


def _coerce_mean_expression(
    regulator_expression: pd.Series | pd.DataFrame | dict[str, float],
) -> pd.Series:
    if isinstance(regulator_expression, pd.Series):
        means = regulator_expression.astype(float).copy()
    elif isinstance(regulator_expression, pd.DataFrame):
        if regulator_expression.empty:
            raise ValueError("regulator_expression dataframe must not be empty")
        if regulator_expression.columns.size > 0:
            means = regulator_expression.astype(float).mean(axis=0)
        else:
            raise ValueError("regulator_expression dataframe must contain gene columns")
    else:
        means = pd.Series(regulator_expression, dtype=float)
    means.index = means.index.astype(str)
    values = means.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("regulator mean expression must be finite")
    if (values < 0).any():
        raise ValueError("regulator mean expression must be non-negative")
    return means


def validate_grn(edges: pd.DataFrame, config: GRNConfig | None = None) -> pd.DataFrame:
    """校验并标准化 GRN 边表。

    输入的 ``edges`` 是 pandas DataFrame。这里会检查必需列、缺失值、
    权重非负、Hill 参数为正，并把 sign 统一成标准字符串。

    返回值仍然是 DataFrame，但列顺序固定，且缺失的 ``hill_coefficient``
    和 ``half_response`` 会用 ``GRNConfig`` 默认值补齐。``threshold`` 是
    旧接口兼容别名，会和 ``half_response`` 保持一致。
    """

    config = config or GRNConfig()
    missing = [col for col in REQUIRED_COLUMNS if col not in edges.columns]
    if missing:
        raise ValueError(f"GRN is missing required columns: {missing}")

    normalized = edges.copy()
    for col in ("regulator", "target"):
        if normalized[col].isna().any():
            raise ValueError(f"GRN column {col!r} contains missing values")
        normalized[col] = normalized[col].astype(str)

    normalized["weight"] = pd.to_numeric(normalized["weight"], errors="raise")
    weights = normalized["weight"].to_numpy(dtype=float)
    if not np.isfinite(weights).all():
        raise ValueError("GRN weights must be finite")
    if (weights < 0).any():
        raise ValueError("GRN weights must be non-negative; sign controls activation/repression")

    normalized["sign"] = normalized["sign"].map(normalize_sign)

    if "hill_coefficient" not in normalized.columns:
        normalized["hill_coefficient"] = config.default_hill_coefficient
    if "half_response" not in normalized.columns and "threshold" in normalized.columns:
        normalized["half_response"] = normalized["threshold"]
    if "threshold" not in normalized.columns and "half_response" in normalized.columns:
        normalized["threshold"] = normalized["half_response"]
    if "half_response" not in normalized.columns:
        normalized["half_response"] = config.default_threshold
    if "threshold" not in normalized.columns:
        normalized["threshold"] = config.default_threshold

    for col in OPTIONAL_COLUMNS:
        normalized[col] = pd.to_numeric(normalized[col], errors="raise")
        values = normalized[col].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"GRN column {col!r} must be finite")
        if col == "hill_coefficient":
            if (values <= 0).any():
                raise ValueError(f"GRN column {col!r} must be positive")
        else:
            if (values < 0).any():
                raise ValueError(f"GRN column {col!r} must be non-negative")

    return normalized[list(RETURN_COLUMNS)]


def calibrate_half_response(
    grn: GRN | pd.DataFrame,
    regulator_expression: pd.Series | pd.DataFrame | dict[str, float],
) -> GRN | pd.DataFrame:
    """Set half-response values from regulator mean expression.

    This mirrors the key SERGIO idea that each edge's half-response is derived
    from the corresponding regulator's mean expression. If a DataFrame is
    provided, means are computed across rows so columns should be gene ids.
    ``threshold`` is kept synchronized as a backward-compatible alias.
    """

    means = _coerce_mean_expression(regulator_expression)
    if isinstance(grn, GRN):
        edges = grn.to_dataframe()
    else:
        edges = validate_grn(grn)

    missing = sorted(set(edges["regulator"]) - set(means.index))
    if missing:
        raise ValueError(f"missing mean expression for regulators: {missing}")

    calibrated = edges.copy()
    calibrated["half_response"] = calibrated["regulator"].map(means).astype(float)
    calibrated["threshold"] = calibrated["half_response"]

    if isinstance(grn, GRN):
        return GRN.from_dataframe(calibrated, genes=grn.genes)
    return calibrated


@dataclass(frozen=True)
class GRN:
    """一个已经校验过的 GRN 对象。

    ``edges`` 保存标准化后的边表；``genes`` 保存全体基因的固定顺序。
    这个顺序非常重要，因为模拟中的矩阵都是按照 ``genes`` 的顺序排列。
    """

    edges: pd.DataFrame
    genes: tuple[str, ...]

    @classmethod
    def from_dataframe(
        cls,
        edges: pd.DataFrame,
        genes: Iterable[str] | None = None,
        config: GRNConfig | None = None,
    ) -> "GRN":
        normalized = validate_grn(edges, config=config)
        if genes is None:
            gene_set = set(normalized["regulator"]).union(normalized["target"])
            ordered_genes = tuple(sorted(gene_set))
        else:
            ordered_genes = tuple(str(gene) for gene in genes)
            unknown = set(normalized["regulator"]).union(normalized["target"]) - set(ordered_genes)
            if unknown:
                raise ValueError(f"GRN contains genes absent from genes list: {sorted(unknown)}")
        return cls(edges=normalized, genes=ordered_genes)

    def to_dataframe(self) -> pd.DataFrame:
        """返回标准化边表的副本，避免外部代码直接修改 GRN 内部状态。"""

        return self.edges.copy()
