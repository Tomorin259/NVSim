"""Benchmark regulator-activity modes on SERGIO's 100-gene DS8 dynamics dataset.

This script keeps the GRN, master-production profile, kinetics, and noise fixed
and only changes how regulator activity is read inside the Hill regulation
frontend:

- spliced: use current s(t)
- unspliced: use current u(t)
- total: use u(t) + s(t)

Outputs are written under ``examples/outputs/alpha/benchmark`` so they do not
overwrite the older SERGIO DS8 runs.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nvsim.grn import GRN, calibrate_half_response
from nvsim.output import to_anndata
from nvsim.production import StateProductionProfile
from nvsim.sergio_io import load_sergio_targets_regs
from nvsim.simulate import simulate_bifurcation


DATASET_DIR = ROOT.parent / "SERGIO" / "data_sets" / "De-noised_100G_3T_300cPerT_dynamics_8_DS8"
TARGETS_FILE = DATASET_DIR / "Interaction_cID_8.txt"
REGS_FILE = DATASET_DIR / "Regs_cID_8.txt"
BMAT_FILE = DATASET_DIR / "bMat_cID8.tab"

OUTPUT_ROOT = ROOT / "examples" / "outputs" / "alpha" / "benchmark" / "sergio_100g_ds8_regulator_activity"

TRUNK_STATE = "bin_0"
BRANCH_STATES = {"branch_0": "bin_1", "branch_1": "bin_2"}
MODES = ("spliced", "unspliced", "total")


def load_sergio_noiseless_mean_expression(dataset_dir: Path) -> pd.Series:
    files = sorted(dataset_dir.glob("simulated_noNoise_*.csv"))
    if not files:
        raise FileNotFoundError(f"no simulated_noNoise files found under {dataset_dir}")

    running_sum: pd.Series | None = None
    for path in files:
        frame = pd.read_csv(path, index_col=0)
        gene_mean = frame.mean(axis=1).astype(float)
        gene_mean.index = gene_mean.index.astype(int).astype(str)
        running_sum = gene_mean if running_sum is None else running_sum.add(gene_mean, fill_value=0.0)
    assert running_sum is not None
    return (running_sum / len(files)).sort_index()


def load_bmat(path: Path) -> list[list[float]]:
    frame = pd.read_csv(path, sep="\t", header=None)
    return frame.astype(float).values.tolist()


def compute_branch_separation(result: dict) -> float:
    true_spliced = np.asarray(result["layers"]["true_spliced"], dtype=float)
    obs = result["obs"]
    x = np.log1p(np.maximum(true_spliced, 0.0))
    centered = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    branches = obs["branch"].astype(str).to_numpy()
    branch0 = coords[branches == "branch_0"]
    branch1 = coords[branches == "branch_1"]
    if branch0.size == 0 or branch1.size == 0:
        return float("nan")
    return float(np.linalg.norm(branch0.mean(axis=0) - branch1.mean(axis=0)))


def summarize_result(result: dict, grn: GRN, mean_expression: pd.Series, mode: str) -> dict:
    obs = result["obs"]
    var = result["var"]
    true_alpha = np.asarray(result["layers"]["true_alpha"], dtype=float)
    true_velocity = np.asarray(result["layers"]["true_velocity"], dtype=float)
    branches = obs["branch"].astype(str).to_numpy()
    branch0 = branches == "branch_0"
    branch1 = branches == "branch_1"
    branch_alpha_gap = float("nan")
    if branch0.any() and branch1.any():
        branch_alpha_gap = float(
            np.mean(np.abs(true_alpha[branch0].mean(axis=0) - true_alpha[branch1].mean(axis=0)))
        )

    return {
        "mode": mode,
        "dataset": DATASET_DIR.name,
        "targets_file": str(TARGETS_FILE),
        "regs_file": str(REGS_FILE),
        "bmat_file": str(BMAT_FILE),
        "sergio_bifurcation_matrix": load_bmat(BMAT_FILE),
        "n_cells": int(obs.shape[0]),
        "n_genes": int(var.shape[0]),
        "n_edges": int(grn.edges.shape[0]),
        "n_master_regulators": int((var["gene_role"] == "master_regulator").sum()),
        "n_targets": int((var["gene_role"] == "target").sum()),
        "branch_counts": {str(k): int(v) for k, v in obs["branch"].value_counts().sort_index().items()},
        "trunk_state": TRUNK_STATE,
        "branch_child_states": dict(BRANCH_STATES),
        "transition_schedule": "step",
        "regulator_activity": mode,
        "capture_rate": float(result["uns"]["simulation_config"]["capture_rate"]),
        "dropout_rate": float(result["uns"]["simulation_config"]["dropout_rate"]),
        "poisson_observed": bool(result["uns"]["simulation_config"]["poisson_observed"]),
        "alpha_max": float(result["uns"]["simulation_config"]["alpha_max"]),
        "half_response_min": float(grn.edges["half_response"].min()),
        "half_response_median": float(grn.edges["half_response"].median()),
        "half_response_max": float(grn.edges["half_response"].max()),
        "mean_expression_min": float(mean_expression.min()),
        "mean_expression_median": float(mean_expression.median()),
        "mean_expression_max": float(mean_expression.max()),
        "mean_true_velocity_abs": float(np.mean(np.abs(true_velocity))),
        "branch_alpha_gap_mean_abs": branch_alpha_gap,
        "true_pca_branch_centroid_distance": compute_branch_separation(result),
    }


def build_benchmark_result(
    regulator_activity: str,
    capture_rate: float = 0.6,
    dropout_rate: float = 0.01,
    poisson_observed: bool = True,
) -> tuple[dict, GRN, pd.Series]:
    sergio_inputs = load_sergio_targets_regs(TARGETS_FILE, REGS_FILE)
    mean_expression = load_sergio_noiseless_mean_expression(DATASET_DIR)
    grn = calibrate_half_response(sergio_inputs.grn, mean_expression)
    profile = StateProductionProfile(sergio_inputs.master_production)

    result = simulate_bifurcation(
        grn,
        n_trunk_cells=300,
        n_branch_cells={"branch_0": 300, "branch_1": 300},
        trunk_time=2.0,
        branch_time=2.0,
        dt=0.05,
        production_profile=profile,
        alpha_source_mode="state_anchor",
        trunk_state=TRUNK_STATE,
        branch_child_states=BRANCH_STATES,
        transition_schedule="step",
        alpha_max=5.0,
        regulator_activity=regulator_activity,
        seed=2088,
        capture_rate=capture_rate,
        poisson_observed=poisson_observed,
        dropout_rate=dropout_rate,
    )
    result["uns"]["simulation_config"]["source_targets_file"] = str(TARGETS_FILE)
    result["uns"]["simulation_config"]["source_regs_file"] = str(REGS_FILE)
    result["uns"]["simulation_config"]["source_bmat_file"] = str(BMAT_FILE)
    result["uns"]["simulation_config"]["source_grn_n_genes"] = len(grn.genes)
    result["uns"]["simulation_config"]["source_grn_n_edges"] = int(grn.edges.shape[0])
    result["uns"]["simulation_config"]["sergio_bifurcation_matrix"] = load_bmat(BMAT_FILE)
    return result, grn, mean_expression


def _mode_dir(mode: str) -> Path:
    return OUTPUT_ROOT / mode


def write_mode_outputs(mode: str) -> dict:
    mode_dir = _mode_dir(mode)
    mode_dir.mkdir(parents=True, exist_ok=True)
    result, grn, mean_expression = build_benchmark_result(mode)

    h5ad_path = mode_dir / f"sergio_100g_ds8_{mode}.h5ad"
    summary_path = mode_dir / "summary.json"
    readme_path = mode_dir / "README.txt"

    adata = to_anndata(result)
    adata.write_h5ad(h5ad_path)

    summary = summarize_result(result, grn, mean_expression, mode)
    summary["output_h5ad"] = str(h5ad_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readme_path.write_text(
        "\n".join(
            [
                "NVSim regulator-activity benchmark on SERGIO 100G DS8.",
                "",
                f"Mode: {mode}",
                f"Targets: {TARGETS_FILE}",
                f"Regs: {REGS_FILE}",
                f"bMat: {BMAT_FILE}",
                f"Output: {h5ad_path.name}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    comparison = {"dataset": DATASET_DIR.name, "modes": {}}
    for mode in MODES:
        comparison["modes"][mode] = write_mode_outputs(mode)
        print(f"completed {mode}")
    comparison_path = OUTPUT_ROOT / "comparison_metrics.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
