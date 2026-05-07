"""Run SERGIO's 100-gene 3-state dynamics DS8 dataset through NVSim.

This follows the current SERGIO-style deterministic path used by NVSim:

1. read SERGIO ``Interaction`` and ``Regs`` files directly;
2. calibrate half-response values from SERGIO ``simulated_noNoise`` means;
3. use SERGIO's master-production bins as NVSim source alpha states;
4. simulate a trunk + two-branch bifurcation with NVSim's ODE solver.

The original SERGIO folder stays read-only. This script only reads:
``Interaction_cID_8.txt``, ``Regs_cID_8.txt``, ``bMat_cID8.tab`` and
``simulated_noNoise_*.csv`` from
``../SERGIO/data_sets/De-noised_100G_3T_300cPerT_dynamics_8_DS8``.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nvsim.grn import calibrate_half_response
from nvsim.output import to_anndata
from nvsim.production import StateProductionProfile
from nvsim.sergio_io import load_sergio_targets_regs
from nvsim.simulate import simulate_bifurcation


DATASET_DIR = ROOT.parent / "SERGIO" / "data_sets" / "De-noised_100G_3T_300cPerT_dynamics_8_DS8"
TARGETS_FILE = DATASET_DIR / "Interaction_cID_8.txt"
REGS_FILE = DATASET_DIR / "Regs_cID_8.txt"
BMAT_FILE = DATASET_DIR / "bMat_cID8.tab"
OUTPUT_DIR = Path(__file__).with_name("outputs") / "sergio_100g_ds8_bifurcation"
OUTPUT_H5AD = OUTPUT_DIR / "sergio_100g_ds8_bifurcation.h5ad"
SUMMARY_JSON = OUTPUT_DIR / "summary.json"
README_TXT = OUTPUT_DIR / "README.txt"

TRUNK_STATE = "bin_0"
BRANCH_STATES = {"branch_0": "bin_1", "branch_1": "bin_2"}


def load_sergio_noiseless_mean_expression(dataset_dir: Path) -> pd.Series:
    """Return per-gene mean expression averaged across SERGIO no-noise replicates."""

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
    """Read SERGIO's bifurcation matrix as a plain nested list for summaries."""

    frame = pd.read_csv(path, sep="\t", header=None)
    return frame.astype(float).values.tolist()


def _write_summary(result: dict, grn_edges: pd.DataFrame, mean_expression: pd.Series) -> dict:
    obs = result["obs"]
    var = result["var"]
    config = result["uns"]["simulation_config"]
    summary = {
        "dataset": "SERGIO De-noised_100G_3T_300cPerT_dynamics_8_DS8",
        "targets_file": str(TARGETS_FILE),
        "regs_file": str(REGS_FILE),
        "bmat_file": str(BMAT_FILE),
        "sergio_bifurcation_matrix": load_bmat(BMAT_FILE),
        "half_response_reference": "SERGIO simulated_noNoise mean expression averaged across replicates",
        "n_reference_replicates": len(list(DATASET_DIR.glob("simulated_noNoise_*.csv"))),
        "n_cells": int(obs.shape[0]),
        "n_genes": int(var.shape[0]),
        "n_edges": int(grn_edges.shape[0]),
        "n_master_regulators": int((var["gene_role"] == "master_regulator").sum()),
        "n_targets": int((var["gene_role"] == "target").sum()),
        "branch_counts": {str(key): int(value) for key, value in obs["branch"].value_counts().sort_index().items()},
        "trunk_production_state": config["trunk_production_state"],
        "branch_production_states": dict(config["branch_production_states"]),
        "capture_rate": config["capture_rate"],
        "dropout_rate": config["dropout_rate"],
        "poisson_observed": bool(config["poisson_observed"]),
        "alpha_max": config["alpha_max"],
        "half_response_min": float(grn_edges["half_response"].min()),
        "half_response_median": float(grn_edges["half_response"].median()),
        "half_response_max": float(grn_edges["half_response"].max()),
        "mean_expression_min": float(mean_expression.min()),
        "mean_expression_median": float(mean_expression.median()),
        "mean_expression_max": float(mean_expression.max()),
        "output_h5ad": str(OUTPUT_H5AD) if OUTPUT_H5AD.exists() else None,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    README_TXT.write_text(
        "\n".join(
            [
                "NVSim SERGIO 100G DS8 deterministic bifurcation output.",
                "",
                f"Input targets: {TARGETS_FILE}",
                f"Input regs: {REGS_FILE}",
                f"Input bMat: {BMAT_FILE}",
                "Half-response calibration: SERGIO simulated_noNoise replicate means",
                f"Source states: trunk={TRUNK_STATE}, branch_0={BRANCH_STATES['branch_0']}, branch_1={BRANCH_STATES['branch_1']}",
                "",
                "Files:",
                f"- {SUMMARY_JSON.name}: run summary and parameter snapshot",
                f"- {OUTPUT_H5AD.name}: AnnData export when anndata is available",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def build_sergio_100g_ds8_bifurcation_result(
    capture_rate: float = 0.6,
    dropout_rate: float = 0.01,
    poisson_observed: bool = True,
):
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
        trunk_production_state=TRUNK_STATE,
        branch_production_states=BRANCH_STATES,
        alpha_max=5.0,
        seed=2088,
        capture_rate=capture_rate,
        poisson_observed=poisson_observed,
        dropout_rate=dropout_rate,
    )
    result["uns"]["simulation_config"]["source_targets_file"] = str(TARGETS_FILE)
    result["uns"]["simulation_config"]["source_regs_file"] = str(REGS_FILE)
    result["uns"]["simulation_config"]["source_bmat_file"] = str(BMAT_FILE)
    result["uns"]["simulation_config"]["half_response_calibration"] = "simulated_noNoise_mean_expression"
    result["uns"]["simulation_config"]["half_response_reference_replicates"] = len(
        list(DATASET_DIR.glob("simulated_noNoise_*.csv"))
    )
    result["uns"]["simulation_config"]["source_grn_n_genes"] = len(grn.genes)
    result["uns"]["simulation_config"]["source_grn_n_edges"] = int(grn.edges.shape[0])
    result["uns"]["simulation_config"]["sergio_bifurcation_matrix"] = load_bmat(BMAT_FILE)
    return result, grn.to_dataframe(), mean_expression


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result, grn_edges, mean_expression = build_sergio_100g_ds8_bifurcation_result()
    try:
        adata = to_anndata(result)
    except ImportError:
        print("anndata is not installed; skipped h5ad export")
    else:
        adata.write_h5ad(OUTPUT_H5AD)
        print(f"saved {OUTPUT_H5AD}")

    summary = _write_summary(result, grn_edges, mean_expression)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
