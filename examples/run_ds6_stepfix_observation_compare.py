#!/usr/bin/env python3
"""Generate clean/noisy DS6 stepfix observation comparisons with compact names."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
import warnings

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.quiver import Quiver
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nvsim.noise import generate_observed_counts

INPUT_H5AD = ROOT / "examples" / "outputs" / "ds6_pt_s3_c300_stepfix" / "sergio_100g_ds6_dynamic_graph_stepfix.h5ad"
OUTPUT_DIR = ROOT / "examples" / "outputs" / "ds6_pt_s3_c300_stepfix" / "obs_compare_tuned"
UTILS_PATH = Path("/mnt/second19T/zhaozelin/velocity/plot/gene_to_use/utils.py")
STATE_ORDER = ["bin_0", "bin_1", "bin_2", "bin_3", "bin_4", "bin_5"]
STATE_COLORS = {
    "bin_0": "#4E79A7",
    "bin_1": "#59A14F",
    "bin_2": "#E15759",
    "bin_3": "#F28E2B",
    "bin_4": "#B07AA1",
    "bin_5": "#76B7B2",
}
NOISE_PARAMS = {
    "capture_model": "poisson_capture",
    "capture_rate": 0.75,
    "capture_efficiency_mode": "cell_lognormal",
    "capture_efficiency_cv": 0.10,
    "poisson": True,
    "dropout_rate": 0.0,
    "seed": 17,
}
warnings.filterwarnings(
    "ignore",
    message="Automatic neighbor calculation is deprecated since scvelo==0.4.0*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message="`neighbors` is deprecated since scvelo==0.4.0*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Automatic computation of PCA is deprecated since scvelo==0.4.0*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message="is_categorical_dtype is deprecated*",
    category=DeprecationWarning,
)

SCV_PARAMS = {
    "min_shared_counts": 0,
    "n_top_genes": 100,
    "log": False,
    "n_pcs": 20,
    "n_neighbors": 12,
}
CLEAN_ARROW_STYLE = {
    "width": 0.0011,
    "headwidth": 3.2,
    "headlength": 3.8,
    "headaxislength": 3.2,
}


def _load_plot_phase_portrait():
    spec = importlib.util.spec_from_file_location("gene_to_use_utils", UTILS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load plotting utils from {UTILS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.plot_phase_portrait


def _base_obs(src: ad.AnnData) -> pd.DataFrame:
    obs = src.obs.copy()
    obs["state"] = pd.Categorical(obs["state"].astype(str), categories=STATE_ORDER, ordered=True)
    obs["bin"] = obs["state"].astype(str)
    return obs


def _new_adata(src: ad.AnnData, *, spliced: np.ndarray, unspliced: np.ndarray, capture_efficiency: np.ndarray | None = None) -> ad.AnnData:
    adata = ad.AnnData(X=np.asarray(spliced, dtype=float).copy())
    adata.obs = _base_obs(src)
    if capture_efficiency is not None:
        adata.obs["capture_efficiency"] = np.asarray(capture_efficiency, dtype=float)
    adata.var = src.var.copy()
    adata.var_names = src.var_names.copy()
    adata.layers["spliced"] = np.asarray(spliced, dtype=float).copy()
    adata.layers["unspliced"] = np.asarray(unspliced, dtype=float).copy()
    adata.layers["true_velocity"] = np.asarray(src.layers["true_velocity"], dtype=float).copy()
    adata.layers["true_velocity_u"] = np.asarray(src.layers["true_velocity_u"], dtype=float).copy()
    adata.uns["state_colors"] = [STATE_COLORS[s] for s in STATE_ORDER]
    return adata


def _prepare_scvelo(adata: ad.AnnData) -> ad.AnnData:
    adata = adata.copy()
    scv.pp.filter_and_normalize(
        adata,
        min_shared_counts=SCV_PARAMS["min_shared_counts"],
        n_top_genes=SCV_PARAMS["n_top_genes"],
        log=SCV_PARAMS["log"],
    )
    scv.pp.moments(adata, n_pcs=SCV_PARAMS["n_pcs"], n_neighbors=SCV_PARAMS["n_neighbors"])
    return adata


def _tune_clean_quiver(ax):
    for artist in ax.collections:
        if isinstance(artist, Quiver):
            artist.width = CLEAN_ARROW_STYLE["width"]
            artist.headwidth = CLEAN_ARROW_STYLE["headwidth"]
            artist.headlength = CLEAN_ARROW_STYLE["headlength"]
            artist.headaxislength = CLEAN_ARROW_STYLE["headaxislength"]


def _plot_phase_pages(adata: ad.AnnData, *, out_prefix: str, tune_clean_arrows: bool, plot_phase_portrait) -> list[str]:
    genes = list(map(str, adata.var_names[:100]))
    page_paths: list[str] = []
    for page_idx in range(4):
        fig, axes = plt.subplots(5, 5, figsize=(22, 22))
        for j in range(25):
            gene_idx = page_idx * 25 + j
            ax = axes.flat[j]
            gene = genes[gene_idx]
            plot_phase_portrait(
                adata,
                gene,
                ax=ax,
                layer_s="Ms",
                layer_u="Mu",
                layer_v_s="true_velocity",
                layer_v_u="true_velocity_u",
                color_key="state",
                s=10,
                alpha=0.65,
                arrow_grid=(30, 30),
                legend_loc=None,
            )
            if tune_clean_arrows:
                _tune_clean_quiver(ax)
        fig.tight_layout()
        out_path = OUTPUT_DIR / f"{out_prefix}_{page_idx + 1}.png"
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        page_paths.append(str(out_path))
    return page_paths


def _prepare_total_umap(src: ad.AnnData, *, total: np.ndarray) -> ad.AnnData:
    adata = ad.AnnData(X=total.copy())
    adata.obs = _base_obs(src)
    adata.var_names = src.var_names.copy()
    adata.uns["state_colors"] = [STATE_COLORS[s] for s in STATE_ORDER]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=SCV_PARAMS["n_pcs"])
    sc.pp.neighbors(adata, n_neighbors=SCV_PARAMS["n_neighbors"], n_pcs=SCV_PARAMS["n_pcs"])
    sc.tl.umap(adata, random_state=17)
    return adata


def _plot_umap(adata: ad.AnnData, *, color: str, title: str, out_name: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    kwargs = {"ax": ax, "show": False, "title": title}
    if color == "state":
        kwargs["palette"] = [STATE_COLORS[s] for s in STATE_ORDER]
    else:
        kwargs["color_map"] = "viridis"
    sc.pl.umap(adata, color=color, **kwargs)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / out_name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _panel_umap(clean_umap: ad.AnnData, noisy_umap: ad.AnnData):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sc.pl.umap(clean_umap, color="state", palette=[STATE_COLORS[s] for s in STATE_ORDER], ax=axes[0, 0], show=False, title="clean total by bin")
    sc.pl.umap(noisy_umap, color="state", palette=[STATE_COLORS[s] for s in STATE_ORDER], ax=axes[0, 1], show=False, title="noisy total by bin")
    sc.pl.umap(clean_umap, color="pseudotime", color_map="viridis", ax=axes[1, 0], show=False, title="clean total by pseudotime")
    sc.pl.umap(noisy_umap, color="pseudotime", color_map="viridis", ax=axes[1, 1], show=False, title="noisy total by pseudotime")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "umap_panel.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _mean_gene_corr(a: np.ndarray, b: np.ndarray) -> float:
    vals = []
    for g in range(a.shape[1]):
        x = a[:, g]
        y = b[:, g]
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        vals.append(np.corrcoef(x, y)[0, 1])
    return float(np.mean(vals))


def _mean_var_ratio(a: np.ndarray, b: np.ndarray) -> float:
    va = np.var(a, axis=0)
    vb = np.var(b, axis=0)
    mask = va > 1e-12
    return float(np.mean(vb[mask] / va[mask]))


def _state_range_ratio(clean: ad.AnnData, noisy: ad.AnnData) -> float:
    states = clean.obs["state"].astype(str).to_numpy()
    tot_c = np.asarray(clean.layers["Mu"]) + np.asarray(clean.layers["Ms"])
    tot_n = np.asarray(noisy.layers["Mu"]) + np.asarray(noisy.layers["Ms"])
    ratios = []
    for g in range(tot_c.shape[1]):
        means_c = []
        means_n = []
        for s in STATE_ORDER:
            mask = states == s
            means_c.append(tot_c[mask, g].mean())
            means_n.append(tot_n[mask, g].mean())
        rc = max(means_c) - min(means_c)
        rn = max(means_n) - min(means_n)
        if rc > 1e-12:
            ratios.append(rn / rc)
    return float(np.mean(ratios))


def _nearest_summary(adata: ad.AnnData) -> dict[str, list[list[object]]]:
    emb = np.asarray(adata.obsm["X_umap"])
    states = adata.obs["state"].astype(str).to_numpy()
    centroids = {s: emb[states == s].mean(axis=0) for s in STATE_ORDER}
    nearest = {}
    for s in STATE_ORDER:
        items = []
        for t in STATE_ORDER:
            if s == t:
                continue
            d = float(np.linalg.norm(centroids[s] - centroids[t]))
            items.append([t, d])
        items.sort(key=lambda x: x[1])
        nearest[s] = items[:3]
    return nearest


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_phase_portrait = _load_plot_phase_portrait()
    src = ad.read_h5ad(INPUT_H5AD)

    clean_raw = _new_adata(
        src,
        spliced=np.asarray(src.layers["true_spliced"]),
        unspliced=np.asarray(src.layers["true_unspliced"]),
    )
    noisy_layers = generate_observed_counts(
        np.asarray(src.layers["true_unspliced"]),
        np.asarray(src.layers["true_spliced"]),
        seed=NOISE_PARAMS["seed"],
        capture_rate=NOISE_PARAMS["capture_rate"],
        poisson=NOISE_PARAMS["poisson"],
        dropout_rate=NOISE_PARAMS["dropout_rate"],
        capture_model=NOISE_PARAMS["capture_model"],
        capture_efficiency_mode=NOISE_PARAMS["capture_efficiency_mode"],
        capture_efficiency_cv=NOISE_PARAMS["capture_efficiency_cv"],
    )
    noisy_raw = _new_adata(
        src,
        spliced=noisy_layers["spliced"],
        unspliced=noisy_layers["unspliced"],
        capture_efficiency=noisy_layers["capture_efficiency"],
    )

    clean = _prepare_scvelo(clean_raw)
    noisy = _prepare_scvelo(noisy_raw)

    clean.write_h5ad(OUTPUT_DIR / "clean_scvelo.h5ad")
    noisy.write_h5ad(OUTPUT_DIR / "noisy_scvelo.h5ad")

    clean_pages = _plot_phase_pages(clean, out_prefix="phase_clean", tune_clean_arrows=True, plot_phase_portrait=plot_phase_portrait)
    noisy_pages = _plot_phase_pages(noisy, out_prefix="phase_noisy", tune_clean_arrows=False, plot_phase_portrait=plot_phase_portrait)

    clean_total = np.asarray(src.layers["true_spliced"]) + np.asarray(src.layers["true_unspliced"])
    noisy_total = np.asarray(noisy_layers["spliced"]) + np.asarray(noisy_layers["unspliced"])
    clean_umap = _prepare_total_umap(src, total=clean_total)
    noisy_umap = _prepare_total_umap(src, total=noisy_total)
    clean_umap.write_h5ad(OUTPUT_DIR / "clean_umap.h5ad")
    noisy_umap.write_h5ad(OUTPUT_DIR / "noisy_umap.h5ad")
    _plot_umap(clean_umap, color="state", title="clean total by bin", out_name="umap_clean_bin.png")
    _plot_umap(noisy_umap, color="state", title="noisy total by bin", out_name="umap_noisy_bin.png")
    _plot_umap(clean_umap, color="pseudotime", title="clean total by pseudotime", out_name="umap_clean_time.png")
    _plot_umap(noisy_umap, color="pseudotime", title="noisy total by pseudotime", out_name="umap_noisy_time.png")
    _panel_umap(clean_umap, noisy_umap)

    summary = {
        "input_h5ad": str(INPUT_H5AD),
        "output_dir": str(OUTPUT_DIR),
        "noise_params": NOISE_PARAMS,
        "scvelo_preprocess": SCV_PARAMS,
        "clean_arrow_style": CLEAN_ARROW_STYLE,
        "state_order": STATE_ORDER,
        "state_colors": STATE_COLORS,
        "files": {
            "clean_scvelo": str(OUTPUT_DIR / "clean_scvelo.h5ad"),
            "noisy_scvelo": str(OUTPUT_DIR / "noisy_scvelo.h5ad"),
            "clean_umap": str(OUTPUT_DIR / "clean_umap.h5ad"),
            "noisy_umap": str(OUTPUT_DIR / "noisy_umap.h5ad"),
            "phase_clean": clean_pages,
            "phase_noisy": noisy_pages,
            "umap": [
                str(OUTPUT_DIR / "umap_clean_bin.png"),
                str(OUTPUT_DIR / "umap_noisy_bin.png"),
                str(OUTPUT_DIR / "umap_clean_time.png"),
                str(OUTPUT_DIR / "umap_noisy_time.png"),
                str(OUTPUT_DIR / "umap_panel.png"),
            ],
        },
        "metrics": {
            "corr_Mu": _mean_gene_corr(np.asarray(clean.layers["Mu"]), np.asarray(noisy.layers["Mu"])),
            "corr_Ms": _mean_gene_corr(np.asarray(clean.layers["Ms"]), np.asarray(noisy.layers["Ms"])),
            "var_ratio_Mu": _mean_var_ratio(np.asarray(clean.layers["Mu"]), np.asarray(noisy.layers["Mu"])),
            "var_ratio_Ms": _mean_var_ratio(np.asarray(clean.layers["Ms"]), np.asarray(noisy.layers["Ms"])),
            "state_range_ratio_total": _state_range_ratio(clean, noisy),
            "umap_nearest_clean": _nearest_summary(clean_umap),
            "umap_nearest_noisy": _nearest_summary(noisy_umap),
        },
        "phase_layers_used": {"spliced": "Ms", "unspliced": "Mu"},
        "velocity_layers_used": {"spliced_velocity": "true_velocity", "unspliced_velocity": "true_velocity_u"},
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUTPUT_DIR / "README.txt").write_text(
        "DS6 stepfix clean/noisy observation comparison\n"
        "- clean/noisy phase portraits use scVelo moments layers Ms/Mu\n"
        "- noisy counts use tuned poisson_capture with cell-specific capture efficiency\n"
        "- clean arrows are intentionally thinner to preserve bin colors\n"
    )
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
