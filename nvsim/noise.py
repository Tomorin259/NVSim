"""从 true layers 生成 observed layers 的轻量噪声模型。

本模块只生成观测层，不会修改 true_unspliced/true_spliced。
当前支持两种 capture 路径：

- ``scale_poisson``: 先做 capture scaling，再可选 Poisson 采样
- ``binomial_capture``: VeloSim-style molecule capture,
  ``Obs ~ Binomial(round(True), capture_rate)``
"""

from __future__ import annotations

import numpy as np


def _resolve_noise_model_name(noise_model: str | None, capture_model: str | None) -> str:
    alias_map = {
        "scale_poisson": "poisson_capture",
        "poisson_capture": "poisson_capture",
        "binomial": "binomial_capture",
        "binomial_capture": "binomial_capture",
    }
    if noise_model is not None:
        if noise_model not in alias_map:
            raise ValueError(
                "noise_model must be one of 'scale_poisson', 'poisson_capture', 'binomial', or 'binomial_capture'"
            )
        return alias_map[noise_model]
    model = "scale_poisson" if capture_model is None else capture_model
    if model not in alias_map:
        raise ValueError(
            "capture_model must be one of 'scale_poisson', 'poisson_capture', 'binomial', or 'binomial_capture'"
        )
    return alias_map[model]


def generate_observed_counts(
    true_unspliced: object,
    true_spliced: object,
    seed: int | None = 0,
    capture_rate: float | None = None,
    poisson: bool = True,
    dropout_rate: float = 0.0,
    noise_model: str | None = None,
    capture_model: str = "scale_poisson",
) -> dict[str, np.ndarray]:
    """根据 true u/s 生成 observed unspliced/spliced。

    ``capture_model='scale_poisson'`` 或 ``noise_model='poisson_capture'`` 时，
    处理顺序是：
    可选 capture scaling -> 可选 Poisson 采样 -> 可选 dropout。

    ``capture_model='binomial_capture'``、``capture_model='binomial'`` 或
    ``noise_model='binomial_capture'`` 时，处理顺序是：
    round(true counts) -> Binomial capture -> 可选 dropout。

    对 binomial capture，当前使用 ``np.rint(true_counts).astype(int)`` 作为
    latent molecule count，然后做按分子捕获。默认不再对 binomial 结果重复
    Poisson 采样，因为 binomial capture 本身已经是离散观测步骤。

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

    resolved_noise_model = _resolve_noise_model_name(noise_model, capture_model)

    if dropout_rate < 0 or dropout_rate > 1:
        raise ValueError("dropout_rate must be in [0, 1]")
    if capture_rate is not None:
        capture_rate_arr = np.asarray(capture_rate, dtype=float)
        if not np.isfinite(capture_rate_arr).all():
            raise ValueError("capture_rate must be finite")
        if ((capture_rate_arr < 0) | (capture_rate_arr > 1)).any():
            raise ValueError("capture_rate must be in [0, 1]")

    if resolved_noise_model == "binomial_capture":
        if capture_rate is None:
            raise ValueError("capture_rate must be provided for noise_model='binomial_capture'")
        observed_u = rng.binomial(np.rint(u).astype(int), capture_rate).astype(float)
        observed_s = rng.binomial(np.rint(s).astype(int), capture_rate).astype(float)
    else:
        if capture_rate is not None:
            u = u * capture_rate
            s = s * capture_rate

        if poisson:
            observed_u = rng.poisson(u).astype(float)
            observed_s = rng.poisson(s).astype(float)
        else:
            observed_u = u.copy()
            observed_s = s.copy()

    if dropout_rate:
        keep_u = rng.binomial(1, 1.0 - dropout_rate, size=observed_u.shape)
        keep_s = rng.binomial(1, 1.0 - dropout_rate, size=observed_s.shape)
        observed_u *= keep_u
        observed_s *= keep_s

    return {"unspliced": observed_u, "spliced": observed_s}
