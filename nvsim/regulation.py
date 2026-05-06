"""GRN 控制 transcription rate alpha 的调控响应函数。

本模块实现从 regulator 表达量到 target transcription rate 的映射：

    regulator spliced expression s_j(t)
        -> Hill activation/repression response
        -> contribution to alpha_i(t)

当前 MVP 使用 additive regulation：多个上游 regulator 的贡献相加。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .grn import GRN, validate_grn


def _as_nonnegative_array(values: object, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return np.maximum(arr, 0.0)


def hill_activation(x: object, half_response: float = 1.0, hill_coefficient: float = 2.0) -> np.ndarray:
    """Hill 激活函数：x^n / (h^n + x^n)。

    x 是 regulator 当前表达量；half_response h 控制半饱和位置；
    hill_coefficient n 控制曲线陡峭程度。返回值在 [0, 1]。
    """

    if half_response <= 0 or hill_coefficient <= 0:
        raise ValueError("half_response and hill_coefficient must be positive")
    x_arr = _as_nonnegative_array(x, "x")
    numerator = np.power(x_arr, hill_coefficient)
    denominator = np.power(half_response, hill_coefficient) + numerator
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0)


def hill_repression(x: object, half_response: float = 1.0, hill_coefficient: float = 2.0) -> np.ndarray:
    """Hill 抑制函数：h^n / (h^n + x^n)。

    regulator 越高，响应越低。注意这里返回的是非负 gate，
    因此 repression contribution 是 ``weight * H_rep(x)``，不是负数。
    """

    return 1.0 - hill_activation(x, half_response=half_response, hill_coefficient=hill_coefficient)


def compute_alpha(
    regulator_values: pd.Series | dict[str, float],
    grn: GRN | pd.DataFrame,
    source_alpha: pd.Series | dict[str, float] | float = 0.0,
    target_leak_alpha: pd.Series | dict[str, float] | float = 0.0,
    master_regulators: list[str] | tuple[str, ...] | pd.Index | None = None,
    alpha_min: float = 0.0,
    alpha_max: float | None = None,
    return_edge_contributions: bool = False,
    basal_alpha: pd.Series | dict[str, float] | float | None = None,
) -> pd.Series | tuple[pd.Series, pd.Series]:
    """计算某一个细胞/时间点的 GRN-controlled alpha。

    ``regulator_values`` 通常是当前时刻每个基因的 spliced RNA ``s(t)``。
    master regulators 的 production rate 直接来自 ``source_alpha``。
    non-master genes 的 alpha 按 SERGIO-style additive Hill contribution 计算：

    - activation: contribution = weight * H_act(s_regulator)
    - repression: contribution = weight * H_rep(s_regulator)

    所有 contribution 加到 non-master target 的 ``target_leak_alpha`` 上，
    最后按 ``alpha_min`` 和可选 ``alpha_max`` 裁剪。
    """

    if alpha_max is not None and alpha_max < alpha_min:
        raise ValueError("alpha_max must be greater than or equal to alpha_min")
    if basal_alpha is not None:
        target_leak_alpha = basal_alpha

    edges = grn.edges if isinstance(grn, GRN) else validate_grn(grn)
    genes = grn.genes if isinstance(grn, GRN) else tuple(sorted(set(edges["regulator"]).union(edges["target"])))
    values = pd.Series(regulator_values, dtype=float).reindex(genes, fill_value=0.0)
    values = values.clip(lower=0.0)
    master_set = {str(gene) for gene in (master_regulators if master_regulators is not None else ())}

    if np.isscalar(source_alpha):
        source = pd.Series(0.0, index=genes, dtype=float)
        if master_set:
            source.loc[list(master_set)] = float(source_alpha)
    else:
        source = pd.Series(source_alpha, dtype=float).reindex(genes, fill_value=0.0)
    source = source.clip(lower=0.0)

    if np.isscalar(target_leak_alpha):
        leak = pd.Series(float(target_leak_alpha), index=genes, dtype=float)
    else:
        leak = pd.Series(target_leak_alpha, dtype=float).reindex(genes, fill_value=0.0)
    leak = leak.clip(lower=0.0)

    alpha = leak.copy()
    if master_set:
        alpha.loc[list(master_set)] = source.loc[list(master_set)]
    edge_contributions = np.zeros(edges.shape[0], dtype=float)

    for idx, edge in enumerate(edges.itertuples(index=False)):
        x = float(values.get(edge.regulator, 0.0))
        if edge.sign == "activation":
            response = hill_activation(x, edge.half_response, edge.hill_coefficient)
        else:
            response = hill_repression(x, edge.half_response, edge.hill_coefficient)
        contribution = float(edge.K) * float(response)
        edge_contributions[idx] = contribution
        if edge.target not in master_set:
            alpha.loc[edge.target] = alpha.get(edge.target, 0.0) + contribution

    alpha = alpha.clip(lower=alpha_min)
    if alpha_max is not None:
        alpha = alpha.clip(upper=alpha_max)
    if return_edge_contributions:
        return alpha, pd.Series(edge_contributions, index=edges.index, name="edge_contribution", dtype=float)
    return alpha
