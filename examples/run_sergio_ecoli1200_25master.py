
"""Run NVSim on SERGIO/GNW Ecoli-1200 with multiple branch-driving masters.

This example keeps the SERGIO source tree read-only and converts the
Ecoli-1200 DOT network into NVSim's GRN schema. Compared with
run_sergio_grn_bifurcation.py, this version drives branch programs through
25 master regulators instead of three so the simulated manifold has a
higher-dimensional GRN-driven signal.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nvsim.output import to_anndata
from nvsim.programs import linear_decrease, linear_increase, sigmoid_decrease, sigmoid_increase
from nvsim.simulate import simulate_bifurcation

from run_sergio_grn_bifurcation import load_sergio_dot_grn, top_master_regulators


N_BRANCH_MASTERS = 25
ECOLI_1200_DOT = ROOT.parent / "SERGIO" / "GNW_sampled_GRNs" / "Ecoli_1200_net4.dot"
OUTPUT_DIR = Path(__file__).with_name("outputs") / "sergio_ecoli1200_25master"


def _trunk_program(index: int):
    """Return a smooth trunk alpha program for a master regulator."""

    family = index % 4
    if family == 0:
        return linear_increase(0.20 + 0.02 * index, 0.70 + 0.03 * index)
    if family == 1:
        return sigmoid_increase(0.25 + 0.01 * index, 0.80 + 0.02 * index, midpoint=0.55)
    if family == 2:
        return linear_decrease(0.90 - 0.01 * index, 0.35 + 0.01 * index)
    return sigmoid_decrease(0.95 - 0.01 * index, 0.30 + 0.01 * index, midpoint=0.55)


def _branch_programs(index: int):
    """Return opposing branch programs for a master regulator."""

    base_low = 0.18 + 0.015 * index
    base_mid = 0.62 + 0.025 * index
    base_high = 1.15 + 0.035 * index
    family = index % 4
    if family == 0:
        return linear_increase(base_mid, base_high), linear_decrease(base_mid, base_low)
    if family == 1:
        return sigmoid_increase(base_mid, base_high, midpoint=0.55), sigmoid_decrease(base_mid, base_low, midpoint=0.55)
    if family == 2:
        return linear_decrease(base_mid, base_low), linear_increase(base_mid, base_high)
    return sigmoid_decrease(base_mid, base_low, midpoint=0.55), sigmoid_increase(base_mid, base_high, midpoint=0.55)


def build_sergio_ecoli1200_25master_result(
    capture_rate: float = 0.6,
    dropout_rate: float = 0.01,
    poisson_observed: bool = True,
    n_branch_masters: int = N_BRANCH_MASTERS,
):
    """Generate an Ecoli-1200 bifurcation dataset with 25 dynamic masters."""

    grn = load_sergio_dot_grn(ECOLI_1200_DOT, seed=2046)
    masters = top_master_regulators(grn, n=n_branch_masters)
    if len(masters) < n_branch_masters:
        raise ValueError(f"expected at least {n_branch_masters} master regulators in the SERGIO GRN")

    master_programs = {gene: _trunk_program(i) for i, gene in enumerate(masters)}
    branch_0 = {}
    branch_1 = {}
    for i, gene in enumerate(masters):
        program_0, program_1 = _branch_programs(i)
        branch_0[gene] = program_0
        branch_1[gene] = program_1

    result = simulate_bifurcation(
        grn,
        n_trunk_cells=300,
        n_branch_cells={"branch_0": 500, "branch_1": 500},
        trunk_time=2.0,
        branch_time=2.5,
        dt=0.05,
        master_programs=master_programs,
        branch_master_programs={"branch_0": branch_0, "branch_1": branch_1},
        alpha_max=4.0,
        seed=2046,
        capture_rate=capture_rate,
        poisson_observed=poisson_observed,
        dropout_rate=dropout_rate,
    )
    result["uns"]["simulation_config"]["source_grn"] = str(ECOLI_1200_DOT)
    result["uns"]["simulation_config"]["source_grn_n_genes"] = len(grn.genes)
    result["uns"]["simulation_config"]["source_grn_n_edges"] = int(grn.edges.shape[0])
    result["uns"]["simulation_config"]["branch_program_master_genes"] = masters
    result["uns"]["simulation_config"]["n_branch_program_master_genes"] = len(masters)
    return result


def main() -> None:
    result = build_sergio_ecoli1200_25master_result()
    grn = result["uns"]["true_grn"]
    summary = {
        "n_cells": result["layers"]["true_spliced"].shape[0],
        "n_genes": result["layers"]["true_spliced"].shape[1],
        "n_edges": int(grn.shape[0]),
        "branches": sorted(result["obs"]["branch"].unique()),
        "source_grn": result["uns"]["simulation_config"]["source_grn"],
        "branch_program_master_genes": result["uns"]["simulation_config"]["branch_program_master_genes"],
    }
    print(summary)
    try:
        adata = to_anndata(result)
    except ImportError:
        print("anndata is not installed; skipped h5ad export")
    else:
        out = OUTPUT_DIR / "sergio_ecoli1200_25master_bifurcation.h5ad"
        out.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(out)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
