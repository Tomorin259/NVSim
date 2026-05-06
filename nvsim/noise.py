"""从 true layers 生成 observed layers 的轻量噪声模型。

本模块只生成观测层，不会修改 true_unspliced/true_spliced。
当前噪声包括 capture rate、Poisson sampling 和 dropout。
"""

from __future__ import annotations

import numpy as np


def generate_observed_counts(
    true_unspliced: object,
    true_spliced: object,
    seed: int | None = 0,
    capture_rate: float | None = None,
    poisson: bool = True,
    dropout_rate: float = 0.0,
) -> dict[str, np.ndarray]:
    """根据 true u/s 生成 observed unspliced/spliced。

    处理顺序是：可选 capture scaling -> 可选 Poisson 采样 -> 可选 dropout。
    如果 ``poisson=False``，observed layer 会保留连续值，适合低噪声可视化
    或调试；它不是现实 UMI count 模型。
    """

    rng = np.random.default_rng(seed)
    u = np.asarray(true_unspliced, dtype=float).copy()
    s = np.asarray(true_spliced, dtype=float).copy()
    if u.shape != s.shape:
        raise ValueError("true_unspliced and true_spliced must have the same shape")
    if (u < 0).any() or (s < 0).any():
        raise ValueError("true values must be non-negative")

    if capture_rate is not None:
        if capture_rate < 0 or capture_rate > 1:
            raise ValueError("capture_rate must be in [0, 1]")
        u = u * capture_rate
        s = s * capture_rate

    if poisson:
        observed_u = rng.poisson(u).astype(float)
        observed_s = rng.poisson(s).astype(float)
    else:
        observed_u = u.copy()
        observed_s = s.copy()

    if dropout_rate:
        if dropout_rate < 0 or dropout_rate > 1:
            raise ValueError("dropout_rate must be in [0, 1]")
        keep_u = rng.binomial(1, 1.0 - dropout_rate, size=observed_u.shape)
        keep_s = rng.binomial(1, 1.0 - dropout_rate, size=observed_s.shape)
        observed_u *= keep_u
        observed_s *= keep_s

    return {"unspliced": observed_u, "spliced": observed_s}
