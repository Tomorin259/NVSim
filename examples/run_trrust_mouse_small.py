#!/usr/bin/env python3
"""Run a first external TRRUST mouse NVSim graph benchmark example."""

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
from nvsim.modes import branching_graph
from nvsim.output import to_anndata
from nvsim.plotting import (
    plot_gene_dynamics,
    plot_phase_portrait,
    plot_phase_gallery,
    plot_showcase,
    prepare_adata,
    select_genes,
)
from nvsim.production import StateProductionProfile
from nvsim.simulate import simulate

DATA_DIR = ROOT / "data" / "external" / "trrust_mouse"
OUTPUT_DIR = ROOT / "examples" / "outputs" / "trrust_mouse_small"
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


def build_trrust_mouse_branching_result(seed: int = 1) -> dict:
    grn, masters, production = load_trrust_small_inputs()
    return simulate(
        grn,
        graph=branching_graph("root", ["branch0", "branch1"]),
        master_regulators=masters,
        production_profile=production,
        alpha_source_mode="state_anchor",
        n_cells_per_state={"root": 70, "branch0": 90, "branch1": 90},
        root_time=1.5,
        state_time={"root": 1.5, "branch0": 2.0, "branch1": 2.0},
        dt=0.02,
        transition_schedule="linear",
        seed=seed,
        capture_rate=None,
        poisson_observed=False,
        dropout_rate=0.0,
        regulator_activity="spliced",
        half_response_calibration="off",
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
    for column in ["pseudotime", "branch", "state"]:
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
    summary = {
        "n_cells": int(obs.shape[0]),
        "n_genes": int(var.shape[0]),
        "states": sorted(obs["state"].astype(str).unique().tolist()) if "state" in obs.columns else [],
        "obs_columns": list(obs.columns),
        "var_columns": list(var.columns),
        "uns_keys": list(uns.keys()),
        "layer_stats": layer_stats,
    }
    return summary, warnings


def save_visualizations(result: dict, output_dir: Path) -> dict[str, object]:
    plots_dir = output_dir / "plots"
    diagnostics_dir = plots_dir / "diagnostics"
    for directory in [plots_dir, diagnostics_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    showcase = plot_showcase(result, output_dir=plots_dir / "velocity_showcase", expression_layer="true", random_state=1)
    adata = prepare_adata(result, expression_layer="true")
    selected_genes = select_genes(adata)
    plot_phase_gallery(result, genes=selected_genes, mode="true", output_path=diagnostics_dir / "phase_gallery_true.png")
    plot_phase_gallery(result, genes=selected_genes, mode="observed", output_path=diagnostics_dir / "phase_gallery_observed.png")
    plot_gene_dynamics(result, selected_genes, output_path=diagnostics_dir / "gene_dynamics_selected.png")
    for gene in selected_genes:
        plot_phase_portrait(result, gene, mode="true", output_path=diagnostics_dir / f"phase_portrait_{gene}_true.png")
    (diagnostics_dir / "selected_genes.txt").write_text("\n".join(selected_genes) + "\n", encoding="utf-8")
    return {"selected_genes": selected_genes, "showcase": showcase}


def main() -> None:
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    result = build_trrust_mouse_branching_result(seed=1)
    qc_summary, qc_warnings = quality_check_result(result)
    viz_summary = save_visualizations(result, output_dir)

    h5ad_path = output_dir / "trrust_mouse_small_branching_graph.h5ad"
    try:
        adata = to_anndata(result)
    except ImportError as exc:
        qc_warnings.append(f"anndata export unavailable: {exc}")
    else:
        adata.write_h5ad(h5ad_path)

    summary = {**qc_summary, "visualization": viz_summary, "warnings": qc_warnings, "output_h5ad": str(h5ad_path) if h5ad_path.exists() else None}
    summary_path = output_dir / "quality_check.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"quality summary: {summary_path}")
    if h5ad_path.exists():
        print(f"saved {h5ad_path}")


if __name__ == "__main__":
    main()
