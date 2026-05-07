"""NVSim 输出组织与 AnnData 导出工具。

内部统一使用 plain Python dict，AnnData 是可选导出格式。
所有 layer 矩阵约定为 cells x genes；time_grid 或 segment_time_courses
才是 timepoints x genes。
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import pandas as pd

from .grn import GRN


def _plain_config(config: Any) -> Any:
    if config is None:
        return {}
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    return config


def _anndata_safe(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
        frame.columns = frame.columns.astype(str)
        frame.index = frame.index.astype(str)
        return {
            "index": frame.index.tolist(),
            "columns": frame.columns.tolist(),
            "data": frame.to_numpy().tolist(),
        }
    if isinstance(value, pd.Series):
        series = value.copy()
        series.index = series.index.astype(str)
        return {"index": series.index.tolist(), "data": series.tolist(), "name": series.name}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _anndata_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_anndata_safe(item) for item in value]
    return value


def _validate_layer_shapes(layers: dict[str, np.ndarray], n_obs: int, n_vars: int) -> None:
    expected = (n_obs, n_vars)
    for name, value in layers.items():
        if value.shape != expected:
            raise ValueError(f"layer {name!r} has shape {value.shape}, expected {expected} (cells x genes)")


def make_result_dict(
    true_unspliced: np.ndarray,
    true_spliced: np.ndarray,
    true_velocity: np.ndarray,
    velocity_u: np.ndarray,
    true_alpha: np.ndarray,
    observed_unspliced: np.ndarray,
    observed_spliced: np.ndarray,
    obs: pd.DataFrame,
    var: pd.DataFrame,
    grn: GRN,
    beta: pd.Series,
    gamma: pd.Series,
    simulation_config: Any = None,
    grn_calibration: Any = None,
    noise_config: Any = None,
    time_grid: pd.DataFrame | None = None,
    edge_contributions: np.ndarray | None = None,
    edge_metadata: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """创建模拟器默认返回的 plain dict。

    layers 中保存 observed 和 true 矩阵：
    ``unspliced/spliced`` 是带噪声观测层；
    ``true_unspliced/true_spliced`` 是真实状态；
    ``true_velocity`` 是 ds/dt；``true_velocity_u`` 是 du/dt；
    ``true_alpha`` 是 GRN 计算得到的 transcription rate。
    """

    layers = {
        "unspliced": np.asarray(observed_unspliced).copy(),
        "spliced": np.asarray(observed_spliced).copy(),
        "true_unspliced": np.asarray(true_unspliced).copy(),
        "true_spliced": np.asarray(true_spliced).copy(),
        "true_velocity": np.asarray(true_velocity).copy(),
        "true_velocity_u": np.asarray(velocity_u).copy(),
        "true_alpha": np.asarray(true_alpha).copy(),
    }
    _validate_layer_shapes(layers, n_obs=obs.shape[0], n_vars=var.shape[0])

    result = {
        "layers": layers,
        "obs": obs.copy(),
        "var": var.copy(),
        "uns": {
            "true_grn": grn.to_dataframe(),
            "kinetic_params": {"beta": beta.copy(), "gamma": gamma.copy()},
            "simulation_config": _plain_config(simulation_config),
            "grn_calibration": _plain_config(grn_calibration),
            "noise_config": _plain_config(noise_config),
        },
        "time_grid": None if time_grid is None else time_grid.copy(),
    }
    if edge_contributions is not None:
        result["edge_contributions"] = np.asarray(edge_contributions).copy()
        result["uns"]["edge_metadata"] = (
            grn.to_dataframe() if edge_metadata is None else edge_metadata.copy()
        )
    return result


def to_anndata(result: dict[str, Any]):
    """把 NVSim result dict 转成 AnnData。

    AnnData 中 ``X`` 默认使用 observed spliced；所有 true/observed 矩阵
    都写入 ``adata.layers``，GRN、kinetic parameters 和 simulation config
    写入 ``adata.uns``。
    """

    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("anndata is not installed; use the plain dictionary output instead") from exc

    layers = result["layers"]
    adata = ad.AnnData(X=layers["spliced"], obs=result["obs"].copy(), var=result["var"].copy())
    for name in (
        "unspliced",
        "spliced",
        "true_unspliced",
        "true_spliced",
        "true_velocity",
        "true_velocity_u",
        "true_alpha",
    ):
        adata.layers[name] = layers[name]
    adata.uns["true_grn"] = _anndata_safe(result["uns"]["true_grn"])
    adata.uns["kinetic_params"] = _anndata_safe(result["uns"]["kinetic_params"])
    adata.uns["simulation_config"] = _anndata_safe(result["uns"]["simulation_config"])
    adata.uns["grn_calibration"] = _anndata_safe(result["uns"]["grn_calibration"])
    adata.uns["noise_config"] = _anndata_safe(result["uns"]["noise_config"])
    if "edge_contributions" in result:
        adata.obsm["edge_contributions"] = np.asarray(result["edge_contributions"]).copy()
        adata.uns["edge_metadata"] = _anndata_safe(result["uns"]["edge_metadata"])
    return adata
