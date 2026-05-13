"""Run a small NVSim path-graph ODE MVP example."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from nvsim.grn import GRN
from nvsim.modes import path_graph
from nvsim.output import to_anndata
from nvsim.production import linear_increase, sigmoid_decrease
from nvsim.simulate import simulate


def build_example_grn() -> GRN:
    genes = [f"g{i}" for i in range(20)]
    edges = []
    for target_idx in range(3, 20):
        reg_idx = target_idx % 3
        edges.append(
            {
                "regulator": f"g{reg_idx}",
                "target": f"g{target_idx}",
                "K": 0.4 + 0.02 * target_idx,
                "sign": "activation" if target_idx % 2 == 0 else "repression",
                "half_response": 0.5,
                "hill_coefficient": 2.0,
            }
        )
    return GRN.from_dataframe(pd.DataFrame(edges), genes=genes)


def main() -> None:
    grn = build_example_grn()
    result = simulate(
        grn,
        graph=path_graph(["early", "late"]),
        n_cells_per_state={"early": 50, "late": 50},
        root_time=2.0,
        state_time={"early": 2.0, "late": 2.0},
        dt=0.02,
        alpha_source_mode="continuous_program",
        master_programs={
            "g0": linear_increase(0.2, 1.2),
            "g1": 0.8,
            "g2": sigmoid_decrease(1.1, 0.2),
        },
        seed=42,
        capture_rate=0.5,
        dropout_rate=0.02,
    )

    try:
        adata = to_anndata(result)
    except ImportError:
        layers = result["layers"]
        print("anndata is not installed; plain dictionary summary:")
        print({
            "n_cells": layers["true_spliced"].shape[0],
            "n_genes": layers["true_spliced"].shape[1],
            "layers": sorted(layers),
            "obs_columns": list(result["obs"].columns),
        })
    else:
        out = Path(__file__).with_name("outputs") / "linear_20gene" / "mvp_linear.h5ad"
        out.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(out)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
