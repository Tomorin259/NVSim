"""Run a small NVSim bifurcation ODE MVP example."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_mvp_linear import build_example_grn

from nvsim.output import to_anndata
from nvsim.production import linear_decrease, linear_increase, sigmoid_decrease, sigmoid_increase
from nvsim.simulate import simulate_bifurcation


def build_bifurcation_result(capture_rate: float = 0.5, dropout_rate: float = 0.02, poisson_observed: bool = True):
    """Generate the toy bifurcation dataset used by examples and tests."""

    grn = build_example_grn()
    return simulate_bifurcation(
        grn,
        n_trunk_cells=50,
        n_branch_cells={"branch_0": 60, "branch_1": 60},
        trunk_time=2.0,
        branch_time=2.5,
        dt=0.02,
        master_programs={
            "g0": linear_increase(0.2, 0.8),
            "g1": 0.75,
            "g2": sigmoid_decrease(1.0, 0.35),
        },
        branch_master_programs={
            "branch_0": {
                "g0": linear_increase(0.8, 1.4),
                "g2": sigmoid_increase(0.35, 1.1, midpoint=0.65),
            },
            "branch_1": {
                "g0": linear_decrease(0.8, 0.2),
                "g2": sigmoid_decrease(0.35, 0.05, midpoint=0.65),
            },
        },
        seed=123,
        capture_rate=capture_rate,
        poisson_observed=poisson_observed,
        dropout_rate=dropout_rate,
    )


def main() -> None:
    result = build_bifurcation_result()
    try:
        adata = to_anndata(result)
    except ImportError:
        layers = result["layers"]
        print("anndata is not installed; plain dictionary summary:")
        print({
            "n_cells": layers["true_spliced"].shape[0],
            "n_genes": layers["true_spliced"].shape[1],
            "branches": sorted(result["obs"]["branch"].unique()),
            "obs_columns": list(result["obs"].columns),
        })
    else:
        out = Path(__file__).with_name("outputs") / "bifurcation_20gene_3master" / "mvp_bifurcation.h5ad"
        out.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(out)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
