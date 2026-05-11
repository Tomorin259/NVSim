"""Run a linear trajectory with continuous master-regulator alpha programs."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_mvp_linear import build_example_grn

from nvsim.output import to_anndata
from nvsim.production import linear_increase, sigmoid_decrease
from nvsim.simulate import simulate_linear


def main() -> None:
    result = simulate_linear(
        build_example_grn(),
        n_cells=100,
        time_end=4.0,
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

    out_dir = Path(__file__).with_name("outputs") / "linear_continuous_program"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        to_anndata(result).write_h5ad(out_dir / "linear_continuous_program.h5ad")
    except ImportError:
        print("anndata is not installed; skipped h5ad export")
    print({
        "alpha_source_mode": result["uns"]["simulation_config"]["alpha_source_mode"],
        "n_cells": result["layers"]["true_spliced"].shape[0],
        "n_genes": result["layers"]["true_spliced"].shape[1],
        "output_dir": str(out_dir),
    })


if __name__ == "__main__":
    main()
