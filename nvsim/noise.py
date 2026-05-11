"""从 true layers 生成 observed layers 的轻量噪声模型。

本模块只生成观测层，不会修改 true_unspliced/true_spliced。
对外推荐只使用两个 canonical capture model：

- ``poisson_capture``: 先做 capture scaling，再可选 Poisson 采样
- ``binomial_capture``: VeloSim-style molecule capture,
  ``Obs ~ Binomial(round(True), capture_rate)``

兼容旧脚本时，仍接受 legacy alias：

- ``scale_poisson`` -> ``poisson_capture``
- ``binomial`` -> ``binomial_capture``
"""

from __future__ import annotations

import numpy as np

CANONICAL_NOISE_MODELS = ("poisson_capture", "binomial_capture")
LEGACY_NOISE_MODEL_ALIASES = {
    "scale_poisson": "poisson_capture",
    "binomial": "binomial_capture",
}
SUPPORTED_NOISE_MODEL_NAMES = (*CANONICAL_NOISE_MODELS, *LEGACY_NOISE_MODEL_ALIASES.keys())


def _valid_capture_model_message() -> str:
    return (
        "capture_model must be one of 'poisson_capture' or 'binomial_capture'; "
        "legacy aliases 'scale_poisson' and 'binomial' are still accepted"
    )


def _resolve_capture_model_name(capture_model: str | None) -> str:
    if capture_model is None:
        return "poisson_capture"
    if capture_model in CANONICAL_NOISE_MODELS:
        return capture_model
    if capture_model in LEGACY_NOISE_MODEL_ALIASES:
        return LEGACY_NOISE_MODEL_ALIASES[capture_model]
    raise ValueError(_valid_capture_model_message())
def generate_observed_counts(
    true_unspliced: object,
    true_spliced: object,
    seed: int | None = 0,
    capture_rate: float | None = None,
    poisson: bool = True,
    dropout_rate: float = 0.0,
    capture_model: str | None = None,
) -> dict[str, np.ndarray]:
    """根据 true u/s 生成 observed unspliced/spliced。

    推荐使用 ``capture_model='poisson_capture'`` 或
    ``capture_model='binomial_capture'``。旧别名 ``scale_poisson``、
    ``binomial`` 仍然兼容。

    ``capture_model='poisson_capture'`` 时，
    处理顺序是：
    可选 capture scaling -> 可选 Poisson 采样 -> 可选 dropout。

    ``capture_model='binomial_capture'`` 时，处理顺序是：
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

    resolved_capture_model = _resolve_capture_model_name(capture_model)

    if dropout_rate < 0 or dropout_rate > 1:
        raise ValueError("dropout_rate must be in [0, 1]")
    if capture_rate is not None:
        capture_rate_arr = np.asarray(capture_rate, dtype=float)
        if not np.isfinite(capture_rate_arr).all():
            raise ValueError("capture_rate must be finite")
        if ((capture_rate_arr < 0) | (capture_rate_arr > 1)).any():
            raise ValueError("capture_rate must be in [0, 1]")

    if resolved_capture_model == "binomial_capture":
        if capture_rate is None:
            raise ValueError("capture_rate must be provided for capture_model='binomial_capture'")
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
