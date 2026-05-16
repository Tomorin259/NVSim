"""Observed-count generation for lightweight capture-noise models."""

from __future__ import annotations

import numpy as np

CANONICAL_CAPTURE_MODELS = ("poisson_capture", "binomial_capture")
CANONICAL_CAPTURE_EFFICIENCY_MODES = ("constant", "cell_lognormal")


def _valid_capture_model_message() -> str:
    return "capture_model must be one of 'poisson_capture' or 'binomial_capture'"


def _valid_capture_efficiency_mode_message() -> str:
    return "capture_efficiency_mode must be one of 'constant' or 'cell_lognormal'"


def _resolve_capture_model_name(capture_model: str | None) -> str:
    if capture_model is None:
        return "poisson_capture"
    if capture_model in CANONICAL_CAPTURE_MODELS:
        return capture_model
    raise ValueError(_valid_capture_model_message())


def _resolve_capture_efficiency_mode(capture_efficiency_mode: str | None) -> str:
    if capture_efficiency_mode is None:
        return "constant"
    if capture_efficiency_mode in CANONICAL_CAPTURE_EFFICIENCY_MODES:
        return capture_efficiency_mode
    raise ValueError(_valid_capture_efficiency_mode_message())


def _validate_capture_rate(capture_rate: float | None) -> np.ndarray | None:
    if capture_rate is None:
        return None
    capture_rate_arr = np.asarray(capture_rate, dtype=float)
    if not np.isfinite(capture_rate_arr).all():
        raise ValueError("capture_rate must be finite")
    if ((capture_rate_arr < 0) | (capture_rate_arr > 1)).any():
        raise ValueError("capture_rate must be in [0, 1]")
    return capture_rate_arr


def _constant_capture_efficiency(n_cells: int, capture_rate_arr: np.ndarray | None) -> np.ndarray:
    if capture_rate_arr is None:
        return np.ones(n_cells, dtype=float)
    if capture_rate_arr.ndim == 0:
        return np.full(n_cells, float(capture_rate_arr), dtype=float)
    if capture_rate_arr.shape == (n_cells,):
        return capture_rate_arr.astype(float, copy=True)
    raise ValueError("capture_rate must be scalar or length n_cells when capture_efficiency_mode='constant'")


def _sample_capture_efficiency(
    n_cells: int,
    *,
    rng: np.random.Generator,
    capture_rate: float | None,
    capture_efficiency_mode: str | None,
    capture_efficiency_cv: float,
) -> np.ndarray:
    if capture_efficiency_cv < 0:
        raise ValueError("capture_efficiency_cv must be non-negative")

    resolved_mode = _resolve_capture_efficiency_mode(capture_efficiency_mode)
    capture_rate_arr = _validate_capture_rate(capture_rate)
    if resolved_mode == "constant":
        return _constant_capture_efficiency(n_cells, capture_rate_arr)

    if capture_rate_arr is not None and capture_rate_arr.ndim != 0:
        raise ValueError("capture_rate must be scalar when capture_efficiency_mode='cell_lognormal'")
    mean_efficiency = 1.0 if capture_rate_arr is None else float(capture_rate_arr)
    if mean_efficiency == 0.0 or capture_efficiency_cv == 0.0:
        return np.full(n_cells, mean_efficiency, dtype=float)

    sigma2 = float(np.log1p(capture_efficiency_cv**2))
    sigma = float(np.sqrt(sigma2))
    mu = float(np.log(mean_efficiency) - 0.5 * sigma2)
    sampled = np.exp(rng.normal(loc=mu, scale=sigma, size=n_cells))
    return np.clip(sampled, 0.0, 1.0)


def generate_observed_counts(
    true_unspliced: object,
    true_spliced: object,
    seed: int | None = 0,
    capture_rate: float | None = None,
    poisson: bool = True,
    dropout_rate: float = 0.0,
    capture_model: str | None = None,
    capture_efficiency_mode: str | None = "constant",
    capture_efficiency_cv: float = 0.0,
) -> dict[str, np.ndarray]:
    """Generate observed unspliced/spliced layers from true latent layers.

    Supported capture models:

    - ``poisson_capture``: optional cell-level capture scaling, then optional Poisson sampling
    - ``binomial_capture``: round latent molecule counts, then apply binomial capture

    For ``binomial_capture``, the implementation uses
    ``np.rint(true_counts).astype(int)`` as the latent molecule count before
    molecule-level capture. The binomial branch does not apply an additional
    Poisson sampling step.

    If ``poisson=False`` under ``poisson_capture``, the returned observed layers
    keep continuous values. This is useful for low-noise visualization and
    debugging, not as a realistic UMI count model.
    """

    rng = np.random.default_rng(seed)
    u = np.asarray(true_unspliced, dtype=float).copy()
    s = np.asarray(true_spliced, dtype=float).copy()
    if u.shape != s.shape:
        raise ValueError("true_unspliced and true_spliced must have the same shape")
    if (u < 0).any() or (s < 0).any():
        raise ValueError("true values must be non-negative")

    resolved_capture_model = _resolve_capture_model_name(capture_model)
    resolved_capture_efficiency_mode = _resolve_capture_efficiency_mode(capture_efficiency_mode)

    if dropout_rate < 0 or dropout_rate > 1:
        raise ValueError("dropout_rate must be in [0, 1]")

    capture_efficiency = _sample_capture_efficiency(
        u.shape[0],
        rng=rng,
        capture_rate=capture_rate,
        capture_efficiency_mode=resolved_capture_efficiency_mode,
        capture_efficiency_cv=capture_efficiency_cv,
    )

    if resolved_capture_model == "binomial_capture":
        if capture_rate is None:
            raise ValueError("capture_rate must be provided for capture_model='binomial_capture'")
        if resolved_capture_efficiency_mode != "constant":
            raise ValueError(
                "capture_efficiency_mode must be 'constant' for capture_model='binomial_capture'"
            )
        observed_u = rng.binomial(np.rint(u).astype(int), capture_efficiency[:, None]).astype(float)
        observed_s = rng.binomial(np.rint(s).astype(int), capture_efficiency[:, None]).astype(float)
    else:
        u = u * capture_efficiency[:, None]
        s = s * capture_efficiency[:, None]

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

    return {
        "unspliced": observed_u,
        "spliced": observed_s,
        "capture_efficiency": capture_efficiency,
    }
