"""Observed-count generation for lightweight capture-noise models."""

from __future__ import annotations

import numpy as np

CANONICAL_CAPTURE_MODELS = ("poisson_capture", "binomial_capture")


def _valid_capture_model_message() -> str:
    return "capture_model must be one of 'poisson_capture' or 'binomial_capture'"


def _resolve_capture_model_name(capture_model: str | None) -> str:
    if capture_model is None:
        return "poisson_capture"
    if capture_model in CANONICAL_CAPTURE_MODELS:
        return capture_model
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
    """Generate observed unspliced/spliced layers from true latent layers.

    Supported capture models:

    - ``poisson_capture``: optional capture scaling, then optional Poisson sampling
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
