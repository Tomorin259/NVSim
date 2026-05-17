#!/usr/bin/env python3
"""Run scVelo dynamical analysis on the tuned noisy DS6 stepfix dataset."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import warnings

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nvsim.plotting import plot_phase_portrait

INPUT_H5AD = ROOT / "examples" / "outputs" / "ds6_pt_s3_c300_stepfix" / "obs_compare_tuned" / "ds6_stepfix_obs_noisy_raw.h5ad"
RESULTS_DIR = ROOT / "examples" / "scvelo" / "results" / "ds6_stepfix_obs_noisy_scvelo_dynamical"
STATE_ORDER = ["bin_0", "bin_1", "bin_2", "bin_3", "bin_4", "bin_5"]
STATE_COLORS = {
    "bin_0": "#4E79A7",
    "bin_1": "#59A14F",
    "bin_2": "#E15759",
    "bin_3": "#F28E2B",
    "bin_4": "#B07AA1",
    "bin_5": "#76B7B2",
}
SCV_DYN_PARAMS = {
    "n_jobs": 8,
    "n_neighbors": 12,
    "n_pcs": 20,
    "random_state": 17,
}

warnings.filterwarnings("ignore", category=DeprecationWarning, module="scvelo")
warnings.filterwarnings("ignore", message="is_categorical_dtype is deprecated*", category=DeprecationWarning)



def _prepare() -> ad.AnnData:
    src = ad.read_h5ad(INPUT_H5AD)
    adata = ad.AnnData(X=np.asarray(src.layers["spliced"], dtype=float).copy())
    adata.obs = src.obs.copy()
    adata.var = src.var.copy()
    adata.var_names = src.var_names.copy()
    adata.layers["spliced"] = np.asarray(src.layers["spliced"], dtype=float).copy()
    adata.layers["unspliced"] = np.asarray(src.layers["unspliced"], dtype=float).copy()
    adata.obs["clusters"] = pd.Categorical(adata.obs["bin"].astype(str), categories=STATE_ORDER, ordered=True)
    adata.obs["bin"] = adata.obs["clusters"].astype(str)
    adata.uns["clusters_colors"] = [STATE_COLORS[s] for s in STATE_ORDER]
    adata.uns["bin_colors"] = [STATE_COLORS[s] for s in STATE_ORDER]
    adata.var["velocity_genes"] = True
    return adata


def _run_dynamical(adata: ad.AnnData) -> ad.AnnData:
    scv.settings.verbosity = 3
    scv.settings.presenter_view = True
    scv.set_figure_params("scvelo")
    scv.pp.filter_and_normalize(adata, min_shared_counts=0, n_top_genes=adata.n_vars, log=False)
    sc.pp.pca(adata, n_comps=SCV_DYN_PARAMS["n_pcs"])
    sc.pp.neighbors(adata, n_neighbors=SCV_DYN_PARAMS["n_neighbors"], n_pcs=SCV_DYN_PARAMS["n_pcs"])
    scv.pp.moments(adata)
    sc.tl.umap(adata, random_state=SCV_DYN_PARAMS["random_state"])
    scv.tl.recover_dynamics(adata, n_jobs=SCV_DYN_PARAMS["n_jobs"])
    scv.tl.velocity(adata, mode="dynamical")
    scv.tl.velocity_graph(adata, n_jobs=SCV_DYN_PARAMS["n_jobs"])
    return adata


def _plot_stream(adata: ad.AnnData) -> str:
    path = RESULTS_DIR / "umap_scvelo_velocity_stream.png"
    scv.pl.velocity_embedding_stream(
        adata,
        basis="umap",
        color="clusters",
        palette=[STATE_COLORS[s] for s in STATE_ORDER],
        show=False,
        legend_loc="right margin",
        title="DS6 noisy stepfix scVelo dynamical stream",
    )
    fig = plt.gcf()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_umap(adata: ad.AnnData) -> list[str]:
    paths = []
    for color, out_name, kwargs in [
        ("clusters", "umap_scvelo_clusters.png", {"palette": [STATE_COLORS[s] for s in STATE_ORDER]}),
        ("pseudotime", "umap_scvelo_pseudotime.png", {"color_map": "viridis"}),
        ("latent_time", "umap_scvelo_latent_time.png", {"color_map": "gnuplot"}),
    ]:
        scv.pl.umap(adata, color=color, show=False, **kwargs)
        fig = plt.gcf()
        out_path = RESULTS_DIR / out_name
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(out_path))
    return paths


def _plot_phase_pages(adata: ad.AnnData) -> list[str]:
    genes = list(map(str, adata.var_names[:100]))
    page_paths = []
    for page_idx in range(4):
        fig, axes = plt.subplots(5, 5, figsize=(22, 22))
        for j in range(25):
            gene = genes[page_idx * 25 + j]
            plot_phase_portrait(
                adata,
                gene,
                ax=axes.flat[j],
                layer_s="Ms",
                layer_u="Mu",
                layer_v_s="velocity",
                layer_v_u="velocity_u",
                color_key="clusters",
                s=10,
                alpha=0.65,
                arrow_grid=(30, 30),
                legend_loc=None,
            )
        fig.tight_layout()
        out_path = RESULTS_DIR / f"phase_scvelo_dynamical_{page_idx + 1}.png"
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        page_paths.append(str(out_path))
    return page_paths


def _summary(adata: ad.AnnData, stream_path: str, umap_paths: list[str], phase_paths: list[str]) -> dict[str, object]:
    fit_cols = [c for c in adata.var.columns if c.startswith("fit_")]
    return {
        "input_h5ad": str(INPUT_H5AD),
        "results_dir": str(RESULTS_DIR),
        "conda_env": "indicators",
        "input_mode": "raw noisy counts",
        "scvelo_version": scv.__version__,
        "scanpy_version": sc.__version__,
        "params": SCV_DYN_PARAMS,
        "files": {
            "stream": stream_path,
            "umap": umap_paths,
            "phase": phase_paths,
            "h5ad": str(RESULTS_DIR / "ds6_stepfix_obs_noisy_scvelo_dynamical.h5ad"),
        },
        "layers": sorted(list(adata.layers.keys())),
        "fit_columns": fit_cols,
        "clusters": STATE_ORDER,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    adata = _prepare()
    adata = _run_dynamical(adata)
    if "latent_time" not in adata.obs:
        scv.tl.latent_time(adata)
    stream_path = _plot_stream(adata)
    umap_paths = _plot_umap(adata)
    phase_paths = _plot_phase_pages(adata)
    out_h5ad = RESULTS_DIR / "ds6_stepfix_obs_noisy_scvelo_dynamical.h5ad"
    adata.write_h5ad(out_h5ad)
    summary = _summary(adata, stream_path, umap_paths, phase_paths)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (RESULTS_DIR / "README.txt").write_text(
        "scVelo dynamical analysis on DS6 stepfix tuned noisy observation dataset\n"
        "- input is ds6_stepfix_obs_noisy_raw.h5ad (raw observed spliced/unspliced counts)\n"
        "- cluster label unified to obs[clusters] from bin\n"
        "- phase portraits use utils.plot_phase_portrait with Ms/Mu and scVelo velocity/velocity_u\n"
    )
    print(RESULTS_DIR)


if __name__ == "__main__":
    main()
