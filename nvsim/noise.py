"""Observed-count generation and observation-layer application helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

CANONICAL_CAPTURE_MODELS = ("poisson_capture", "binomial_capture")
CANONICAL_CAPTURE_EFFICIENCY_MODES = ("constant", "cell_lognormal")
CANONICAL_COUNT_MODELS = ("poisson", "binomial")
CANONICAL_CELL_CAPTURE_MODES = ("constant", "lognormal")
CANONICAL_DROPOUT_MODES = ("off", "bernoulli")


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


def _resolve_count_model(count_model: str | None) -> str:
    if count_model is None:
        return "poisson"
    if count_model in CANONICAL_COUNT_MODELS:
        return count_model
    raise ValueError("count_model must be one of 'poisson' or 'binomial'")


def _resolve_cell_capture_mode(cell_capture_mode: str | None) -> str:
    if cell_capture_mode is None:
        return "constant"
    if cell_capture_mode in CANONICAL_CELL_CAPTURE_MODES:
        return cell_capture_mode
    raise ValueError("cell_capture_mode must be one of 'constant' or 'lognormal'")


def _resolve_dropout_mode(dropout_mode: str | None) -> str:
    if dropout_mode is None:
        return "off"
    if dropout_mode in CANONICAL_DROPOUT_MODES:
        return dropout_mode
    raise ValueError("dropout_mode must be one of 'off' or 'bernoulli'")


def _canonical_capture_model_from_count_model(count_model: str) -> str:
    if count_model == "poisson":
        return "poisson_capture"
    if count_model == "binomial":
        return "binomial_capture"
    raise ValueError(f"unsupported count_model {count_model!r}")


def _canonical_capture_efficiency_mode_from_cell_capture_mode(cell_capture_mode: str) -> str:
    if cell_capture_mode == "constant":
        return "constant"
    if cell_capture_mode == "lognormal":
        return "cell_lognormal"
    raise ValueError(f"unsupported cell_capture_mode {cell_capture_mode!r}")


def _resolve_observation_kwargs(
    *,
    count_model: str | None,
    cell_capture_mode: str | None,
    cell_capture_mean: float | None,
    cell_capture_cv: float,
    dropout_mode: str | None,
    dropout_rate: float,
    observation_sample: bool,
    capture_model: str | None,
    capture_rate: float | None,
    capture_efficiency_mode: str | None,
    capture_efficiency_cv: float | None,
    poisson: bool | None,
) -> dict[str, Any]:
    if capture_model is not None:
        count_model = {
            "poisson_capture": "poisson",
            "binomial_capture": "binomial",
        }[_resolve_capture_model_name(capture_model)]
    resolved_count_model = _resolve_count_model(count_model)

    if capture_efficiency_mode is not None:
        cell_capture_mode = {
            "constant": "constant",
            "cell_lognormal": "lognormal",
        }[_resolve_capture_efficiency_mode(capture_efficiency_mode)]
    resolved_cell_capture_mode = _resolve_cell_capture_mode(cell_capture_mode)

    if capture_rate is not None:
        cell_capture_mean = capture_rate
    if capture_efficiency_cv is not None:
        cell_capture_cv = capture_efficiency_cv
    if poisson is not None:
        observation_sample = bool(poisson)
    if dropout_mode is None:
        dropout_mode = "off" if dropout_rate == 0.0 else "bernoulli"
    resolved_dropout_mode = _resolve_dropout_mode(dropout_mode)

    return {
        "count_model": resolved_count_model,
        "cell_capture_mode": resolved_cell_capture_mode,
        "cell_capture_mean": cell_capture_mean,
        "cell_capture_cv": float(cell_capture_cv),
        "dropout_mode": resolved_dropout_mode,
        "dropout_rate": float(dropout_rate),
        "observation_sample": bool(observation_sample),
        "capture_model": _canonical_capture_model_from_count_model(resolved_count_model),
        "capture_efficiency_mode": _canonical_capture_efficiency_mode_from_cell_capture_mode(resolved_cell_capture_mode),
    }


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
    """Generate observed unspliced/spliced layers from true latent layers."""

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


def _copy_result_dict(data: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in data.items():
        if key == "layers":
            copied[key] = {name: np.asarray(layer).copy() for name, layer in value.items()}
        elif key == "obs":
            copied[key] = value.copy()
        elif key == "var":
            copied[key] = value.copy()
        elif key == "uns":
            copied[key] = dict(value)
        elif isinstance(value, np.ndarray):
            copied[key] = value.copy()
        elif hasattr(value, "copy"):
            copied[key] = value.copy()
        else:
            copied[key] = value
    if "uns" not in copied:
        copied["uns"] = {}
    return copied


def _build_observation_config(*, seed: int | None, resolved: dict[str, Any]) -> dict[str, Any]:
    return {
        "count_model": resolved["count_model"],
        "cell_capture_mode": resolved["cell_capture_mode"],
        "cell_capture_mean": resolved["cell_capture_mean"],
        "cell_capture_cv": resolved["cell_capture_cv"],
        "observation_sample": resolved["observation_sample"],
        "dropout_mode": resolved["dropout_mode"],
        "dropout_rate": resolved["dropout_rate"],
        "seed": seed,
    }


def _build_noise_config(observation_config: dict[str, Any], resolved: dict[str, Any]) -> dict[str, Any]:
    return {
        "capture_model": resolved["capture_model"],
        "capture_rate": observation_config["cell_capture_mean"],
        "capture_efficiency_mode": resolved["capture_efficiency_mode"],
        "capture_efficiency_cv": observation_config["cell_capture_cv"],
        "poisson_observed": observation_config["observation_sample"],
        "dropout_rate": observation_config["dropout_rate"],
    }


def apply_observation(
    data: Any,
    *,
    count_model: str | None = "poisson",
    cell_capture_mode: str | None = "constant",
    cell_capture_mean: float | None = None,
    cell_capture_cv: float = 0.0,
    gene_dispersion_mode: str = "off",
    gene_dispersion_strength: float = 0.0,
    dropout_mode: str | None = "off",
    dropout_rate: float = 0.0,
    observation_sample: bool = True,
    seed: int | None = 0,
    capture_model: str | None = None,
    capture_rate: float | None = None,
    capture_efficiency_mode: str | None = None,
    capture_efficiency_cv: float | None = None,
    poisson: bool | None = None,
):
    """Apply the observation model to a clean simulation result or AnnData."""

    if gene_dispersion_mode != "off":
        raise NotImplementedError("gene_dispersion_mode is reserved for the next observation-layer step")
    if gene_dispersion_strength != 0.0:
        raise NotImplementedError("gene_dispersion_strength is reserved for the next observation-layer step")

    resolved = _resolve_observation_kwargs(
        count_model=count_model,
        cell_capture_mode=cell_capture_mode,
        cell_capture_mean=cell_capture_mean,
        cell_capture_cv=cell_capture_cv,
        dropout_mode=dropout_mode,
        dropout_rate=dropout_rate,
        observation_sample=observation_sample,
        capture_model=capture_model,
        capture_rate=capture_rate,
        capture_efficiency_mode=capture_efficiency_mode,
        capture_efficiency_cv=capture_efficiency_cv,
        poisson=poisson,
    )
    observation_config = _build_observation_config(seed=seed, resolved=resolved)
    noise_config = _build_noise_config(observation_config, resolved)

    try:
        import anndata as ad
    except ImportError:
        ad = None

    if ad is not None and isinstance(data, ad.AnnData):
        if "true_unspliced" not in data.layers or "true_spliced" not in data.layers:
            raise ValueError("AnnData input must contain true_unspliced and true_spliced layers")
        observed = generate_observed_counts(
            data.layers["true_unspliced"],
            data.layers["true_spliced"],
            seed=seed,
            capture_rate=observation_config["cell_capture_mean"],
            poisson=observation_config["observation_sample"],
            dropout_rate=observation_config["dropout_rate"],
            capture_model=resolved["capture_model"],
            capture_efficiency_mode=resolved["capture_efficiency_mode"],
            capture_efficiency_cv=observation_config["cell_capture_cv"],
        )
        out = data.copy()
        out.layers["unspliced"] = observed["unspliced"]
        out.layers["spliced"] = observed["spliced"]
        out.obs = out.obs.copy()
        out.obs["capture_efficiency"] = observed["capture_efficiency"]
        out.uns["observation_config"] = observation_config
        out.uns["noise_config"] = noise_config
        out.X = np.asarray(observed["spliced"], dtype=float).copy()
        return out

    if not isinstance(data, dict):
        raise TypeError("apply_observation expects an NVSim result dict or AnnData input")
    if "layers" not in data or "obs" not in data or "uns" not in data:
        raise ValueError("result dict input must contain layers, obs, and uns")
    layers = data["layers"]
    if "true_unspliced" not in layers or "true_spliced" not in layers:
        raise ValueError("result dict input must contain true_unspliced and true_spliced layers")

    observed = generate_observed_counts(
        layers["true_unspliced"],
        layers["true_spliced"],
        seed=seed,
        capture_rate=observation_config["cell_capture_mean"],
        poisson=observation_config["observation_sample"],
        dropout_rate=observation_config["dropout_rate"],
        capture_model=resolved["capture_model"],
        capture_efficiency_mode=resolved["capture_efficiency_mode"],
        capture_efficiency_cv=observation_config["cell_capture_cv"],
    )

    out = _copy_result_dict(data)
    out["layers"]["unspliced"] = observed["unspliced"]
    out["layers"]["spliced"] = observed["spliced"]
    out["obs"] = out["obs"].copy()
    out["obs"]["capture_efficiency"] = observed["capture_efficiency"]
    out["uns"]["observation_config"] = observation_config
    out["uns"]["noise_config"] = noise_config
    return out
