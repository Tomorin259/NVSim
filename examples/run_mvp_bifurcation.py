"""Run a small NVSim branching-graph ODE MVP example."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from run_mvp_linear import build_example_grn

from nvsim.modes import branching_graph
from nvsim.output import to_anndata
from nvsim.production import StateProductionProfile
from nvsim.simulate import simulate


def build_bifurcation_result(capture_rate: float = 0.5, dropout_rate: float = 0.02, poisson_observed: bool = True):
    grn = build_example_grn()
    profile = StateProductionProfile(
        pd.DataFrame(
            {
                "g0": [0.2, 1.4, 0.2],
                "g1": [0.75, 0.75, 0.75],
                "g2": [1.0, 1.1, 0.05],
            },
            index=["root", "branch_0", "branch_1"],
        )
    )
    return simulate(
        grn,
        graph=branching_graph("root", ["branch_0", "branch_1"]),
        production_profile=profile,
        alpha_source_mode="state_anchor",
        n_cells_per_state={"root": 50, "branch_0": 60, "branch_1": 60},
        root_time=2.0,
        state_time={"root": 2.0, "branch_0": 2.5, "branch_1": 2.5},
        dt=0.02,
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
            "states": sorted(result["obs"]["state"].unique()),
            "obs_columns": list(result["obs"].columns),
        })
    else:
        out = Path(__file__).with_name("outputs") / "bifurcation_20gene_3master" / "mvp_bifurcation.h5ad"
        out.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(out)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
