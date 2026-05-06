"""RNA velocity ODE 的动力学参数与初始状态工具。

本模块不处理 GRN，只负责 gene-specific kinetic parameters：

- beta_i: unspliced -> spliced 的 splicing rate
- gamma_i: spliced RNA degradation rate
- u0_i, s0_i: ODE 初始状态

当前 MVP 中 beta/gamma 是基因特异但不随细胞或分支变化。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _gene_index(genes: list[str] | tuple[str, ...]) -> pd.Index:
    if not genes:
        raise ValueError("genes must not be empty")
    return pd.Index([str(gene) for gene in genes], name="gene")


def validate_positive_vector(values: object, genes: list[str] | tuple[str, ...], name: str) -> pd.Series:
    """校验一个按 gene 排列的正值向量。

    用于 beta/gamma 这类必须为正的参数。可以输入 pandas Series
    或 numpy/list；如果是 Series，会按 ``genes`` 重新排序。
    """

    index = _gene_index(genes)
    if isinstance(values, pd.Series):
        series = values.astype(float).reindex(index)
        if series.isna().any():
            missing = list(series[series.isna()].index)
            raise ValueError(f"{name} is missing genes: {missing}")
    else:
        arr = np.asarray(values, dtype=float)
        if arr.shape != (len(index),):
            raise ValueError(f"{name} must have shape ({len(index)},)")
        series = pd.Series(arr, index=index, name=name)

    arr = series.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    if (arr <= 0).any():
        raise ValueError(f"{name} must be positive")
    series.name = name
    return series


def create_kinetic_vectors(
    genes: list[str] | tuple[str, ...],
    beta: object | None = None,
    gamma: object | None = None,
    beta_range: tuple[float, float] = (0.4, 1.2),
    gamma_range: tuple[float, float] = (0.2, 0.8),
    seed: int | None = 0,
) -> tuple[pd.Series, pd.Series]:
    """生成或校验 beta/gamma 向量。

    如果用户没有提供 beta/gamma，就用确定性随机种子从给定范围均匀采样；
    如果用户提供，则只做形状、有限值和正值校验。
    """

    index = _gene_index(genes)
    rng = np.random.default_rng(seed)

    if beta is None:
        lo, hi = beta_range
        if lo <= 0 or hi <= lo:
            raise ValueError("beta_range must be positive and increasing")
        beta_series = pd.Series(rng.uniform(lo, hi, size=len(index)), index=index, name="beta")
    else:
        beta_series = validate_positive_vector(beta, tuple(index), "beta")

    if gamma is None:
        lo, hi = gamma_range
        if lo <= 0 or hi <= lo:
            raise ValueError("gamma_range must be positive and increasing")
        gamma_series = pd.Series(rng.uniform(lo, hi, size=len(index)), index=index, name="gamma")
    else:
        gamma_series = validate_positive_vector(gamma, tuple(index), "gamma")

    return beta_series, gamma_series


def initialize_state(
    genes: list[str] | tuple[str, ...],
    u0: object | None = None,
    s0: object | None = None,
    low: float = 0.0,
    high: float = 0.05,
    seed: int | None = 0,
) -> tuple[pd.Series, pd.Series]:
    """生成或校验 ODE 初始状态 u0/s0。

    初始 unspliced/spliced 状态必须非负。默认给一个很小的随机初始值，
    避免所有基因完全从 0 开始时图形过于退化。
    """

    index = _gene_index(genes)
    rng = np.random.default_rng(seed)

    def build(values: object | None, name: str) -> pd.Series:
        if values is None:
            if low < 0 or high < low:
                raise ValueError("initial-state range must be non-negative and increasing")
            return pd.Series(rng.uniform(low, high, size=len(index)), index=index, name=name)
        if isinstance(values, pd.Series):
            series = values.astype(float).reindex(index)
            if series.isna().any():
                missing = list(series[series.isna()].index)
                raise ValueError(f"{name} is missing genes: {missing}")
        else:
            arr = np.asarray(values, dtype=float)
            if arr.shape != (len(index),):
                raise ValueError(f"{name} must have shape ({len(index)},)")
            series = pd.Series(arr, index=index, name=name)
        arr = series.to_numpy(dtype=float)
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must be finite")
        if (arr < 0).any():
            raise ValueError(f"{name} must be non-negative")
        series.name = name
        return series

    return build(u0, "u0"), build(s0, "s0")
