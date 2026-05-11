#!/usr/bin/env python3
"""Run a first external TRRUST mouse NVSim benchmark example."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nvsim.grn import GRN
from nvsim.output import to_anndata
from nvsim.plotting import (
    compute_umap_embedding,
    plot_embedding_by_branch,
    plot_embedding_by_pseudotime,
    plot_embedding_with_velocity,
    plot_gene_dynamics_over_pseudotime,
    plot_phase_portrait,
    plot_phase_portrait_gallery,
    select_representative_genes_by_dynamics,
)
from nvsim.production import StateProductionProfile
from nvsim.simulate import simulate_bifurcation

DATA_DIR = ROOT / "data" / "external" / "trrust_mouse"
OUTPUT_DIR = ROOT / "examples" / "outputs" / "trrust_mouse_small"
ENABLE_UMAP = False
GRN_COLUMNS = ["regulator", "target", "sign", "K", "half_response", "hill_coefficient"]


def load_trrust_small_inputs(data_dir: Path = DATA_DIR) -> tuple[GRN, list[str], StateProductionProfile]:
    grn_path = data_dir / "trrust_mouse_small_grn.csv"
    masters_path = data_dir / "trrust_mouse_master_regulators.txt"
    production_path = data_dir / "trrust_mouse_mr_production.csv"
    if not grn_path.exists():
        raise FileNotFoundError(f"missing GRN CSV: {grn_path}")
    if not masters_path.exists():
        raise FileNotFoundError(f"missing master regulator file: {masters_path}")
    if not production_path.exists():
        raise FileNotFoundError(f"missing production profile CSV: {production_path}")

    grn_df = pd.read_csv(grn_path)
    masters = [line.strip() for line in masters_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    production_df = pd.read_csv(production_path)
    if "state" not in production_df.columns:
        raise ValueError("master production CSV must contain a state column")
    production = StateProductionProfile(production_df.set_index("state"))
    genes = sorted(set(grn_df["regulator"].astype(str)).union(grn_df["target"].astype(str)))
    grn = GRN.from_dataframe(grn_df.loc[:, GRN_COLUMNS], genes=genes, master_regulators=masters)
    production.validate_master_genes(masters)
    production.validate_states(["root", "branch0", "branch1"])
    return grn, masters, production


def build_trrust_mouse_bifurcation_result(seed: int = 1) -> dict:
    grn, masters, production = load_trrust_small_inputs()
    return simulate_bifurcation(
        grn,
        n_trunk_cells=70,
        n_branch_cells={"branch_0": 90, "branch_1": 90},
        trunk_time=1.5,
        branch_time=2.0,
        dt=0.02,
        master_regulators=masters,
        production_profile=production,
        alpha_source_mode="state_anchor",
        trunk_state="root",
        branch_child_states={"branch_0": "branch0", "branch_1": "branch1"},
        transition_schedule="linear",
        seed=seed,
        capture_rate=None,
        poisson_observed=False,
        dropout_rate=0.0,
        regulator_activity="spliced",
        auto_calibrate_half_response=False,
    )


def _layer_stats(matrix: np.ndarray) -> dict[str, float | int | bool]:
    arr = np.asarray(matrix, dtype=float)
    finite = np.isfinite(arr)
    return {
        "min": float(np.nanmin(arr)) if arr.size else 0.0,
        "max": float(np.nanmax(arr)) if arr.size else 0.0,
        "mean": float(np.nanmean(arr)) if arr.size else 0.0,
        "nonzero": int(np.count_nonzero(arr)),
        "all_zero": bool(np.count_nonzero(arr) == 0),
        "has_nan": bool(np.isnan(arr).any()),
        "has_inf": bool(np.isinf(arr).any()),
        "all_finite": bool(finite.all()),
    }


def quality_check_result(result: dict) -> tuple[dict[str, object], list[str]]:
    warnings: list[str] = []
    layers = result["layers"]
    required_layers = ["spliced", "unspliced", "true_spliced", "true_unspliced", "true_velocity", "true_alpha"]
    for layer_name in required_layers:
        if layer_name not in layers:
            warnings.append(f"missing required layer: {layer_name}")
    obs = result["obs"]
    var = result["var"]
    uns = result["uns"]
    for column in ["pseudotime", "branch"]:
        if column not in obs.columns:
            warnings.append(f"obs is missing column: {column}")
    for column in ["gene_role", "gene_class"]:
        if column not in var.columns:
            warnings.append(f"var is missing column: {column}")
    for key in ["true_grn", "kinetic_params", "simulation_config"]:
        if key not in uns:
            warnings.append(f"uns is missing key: {key}")

    layer_stats = {
        name: _layer_stats(layers[name])
        for name in ["true_alpha", "true_unspliced", "true_spliced", "true_velocity", "unspliced", "spliced"]
        if name in layers
    }
    for name, stats in layer_stats.items():
        if stats["has_nan"] or stats["has_inf"]:
            warnings.append(f"layer {name} contains non-finite values")
        if stats["all_zero"]:
            warnings.append(f"layer {name} is all zero")
        if abs(stats["max"]) > 1e6:
            warnings.append(f"layer {name} has very large values: max={stats['max']}")

    summary = {
        "n_cells": int(obs.shape[0]),
        "n_genes": int(var.shape[0]),
        "branches": sorted(obs["branch"].astype(str).unique().tolist()) if "branch" in obs.columns else [],
        "obs_columns": list(obs.columns),
        "var_columns": list(var.columns),
        "uns_keys": list(uns.keys()),
        "layer_stats": layer_stats,
    }
    return summary, warnings


def save_visualizations(result: dict, output_dir: Path) -> dict[str, object]:
    plots_dir = output_dir / "plots"
    true_dir = plots_dir / "true"
    observed_dir = plots_dir / "observed"
    diagnostics_dir = plots_dir / "diagnostics"
    for directory in [true_dir, observed_dir, diagnostics_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    selection = select_representative_genes_by_dynamics(result, result["uns"]["true_grn"])
    selected_genes = list(dict.fromkeys(selection["genes"].values()))
    plot_embedding_by_pseudotime(result, method="pca", layer_preference="true", output_path=true_dir / "embedding_pca_true_by_pseudotime.png")
    plot_embedding_by_branch(result, method="pca", layer_preference="true", output_path=true_dir / "embedding_pca_true_by_branch.png")
    plot_embedding_with_velocity(result, method="pca", layer_preference="true", output_path=true_dir / "embedding_pca_true_with_velocity.png")
    plot_embedding_by_pseudotime(result, method="pca", layer_preference="observed", output_path=observed_dir / "embedding_pca_observed_by_pseudotime.png")
    plot_embedding_by_branch(result, method="pca", layer_preference="observed", output_path=observed_dir / "embedding_pca_observed_by_branch.png")
    method_used = "disabled"
    if ENABLE_UMAP:
        embedding, method_used, _ = compute_umap_embedding(result, random_state=1, layer_preference="observed")
        if method_used == "umap":
            plot_embedding_by_pseudotime(result, embedding=embedding, method="umap", layer_preference="observed", output_path=observed_dir / "embedding_umap_observed_by_pseudotime.png")
            plot_embedding_by_branch(result, embedding=embedding, method="umap", layer_preference="observed", output_path=observed_dir / "embedding_umap_observed_by_branch.png")
    plot_phase_portrait_gallery(result, genes=selected_genes, mode="true", output_path=true_dir / "phase_portrait_gallery_true.png")
    plot_phase_portrait_gallery(result, genes=selected_genes, mode="observed", output_path=observed_dir / "phase_portrait_gallery_observed.png")
    for label, gene in selection["genes"].items():
        plot_phase_portrait(result, gene, mode="true", output_path=true_dir / f"phase_portrait_{label}_{gene}_true.png")
        plot_gene_dynamics_over_pseudotime(result, gene, include_velocity_u=True, output_path=diagnostics_dir / f"gene_dynamics_{label}_{gene}.png")
    selected_lines = []
    for label, gene in selection["genes"].items():
        selected_lines.append(f"{label}: {gene}")
        selected_lines.append(f"  alpha_difference: {selection['alpha_differences'][label]:.6g}")
    (diagnostics_dir / "selected_genes.txt").write_text("\n".join(selected_lines) + "\n", encoding="utf-8")
    return {
        "selected_genes": selection["genes"],
        "alpha_differences": selection["alpha_differences"],
        "umap_available": bool(method_used == "umap"),
    }


def main() -> None:
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    result = build_trrust_mouse_bifurcation_result(seed=1)
    qc_summary, qc_warnings = quality_check_result(result)
    viz_summary = save_visualizations(result, output_dir)

    h5ad_path = output_dir / "trrust_mouse_small_bifurcation.h5ad"
    try:
        adata = to_anndata(result)
    except ImportError as exc:
        qc_warnings.append(f"anndata export unavailable: {exc}")
    else:
        adata.write_h5ad(h5ad_path)

    summary = {
        **qc_summary,
        "visualization": viz_summary,
        "warnings": qc_warnings,
        "output_h5ad": str(h5ad_path) if h5ad_path.exists() else None,
    }
    summary_path = output_dir / "quality_check.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for name in ["true_alpha", "true_spliced", "true_unspliced", "true_velocity"]:
        if name in qc_summary["layer_stats"]:
            print(name, qc_summary["layer_stats"][name])
    if qc_warnings:
        for warning in qc_warnings:
            print(f"WARNING: {warning}")
    print(f"quality summary: {summary_path}")
    if h5ad_path.exists():
        print(f"saved {h5ad_path}")
    else:
        print("h5ad was not written; see warnings above")


if __name__ == "__main__":
    main()
