"""Canonical NVSim tutorial for the current public simulation API.

This file is the single best place to copy example calls from.
If the public API changes, update this tutorial in the same change set.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nvsim.grn import GRN
from nvsim.output import to_anndata
from nvsim.production import StateProductionProfile, linear_increase, sigmoid_decrease
from nvsim.simulate import simulate_bifurcation, simulate_linear


def build_tutorial_grn() -> GRN:
    """Create a minimal GRN using the canonical edge schema."""

    edges = pd.DataFrame(
        {
            "regulator": ["g0", "g1", "g2"],
            "target": ["g3", "g4", "g5"],
            "sign": ["activation", "activation", "repression"],
            "K": [0.8, 0.6, 0.5],
            "half_response": [0.5, 0.6, 0.5],
            "hill_coefficient": [2.0, 2.0, 2.0],
        }
    )
    genes = [f"g{i}" for i in range(6)]
    masters = ["g0", "g1", "g2"]
    return GRN.from_dataframe(edges, genes=genes, master_regulators=masters)


def build_tutorial_profile() -> StateProductionProfile:
    """Build a small state-anchor production profile."""

    return StateProductionProfile(
        pd.DataFrame(
            {
                "g0": [0.4, 1.2, 0.2],
                "g1": [0.8, 0.4, 1.1],
                "g2": [0.7, 1.0, 0.3],
            },
            index=["root", "branch_A", "branch_B"],
        )
    )


def linear_parameters() -> dict[str, object]:
    """Recommended parameter block for a simple linear tutorial run."""

    return {
        "n_cells": 60,
        "time_end": 3.0,
        "dt": 0.05,
        "alpha_source_mode": "continuous_program",
        "master_programs": {
            "g0": linear_increase(0.2, 1.0),
            "g1": 0.8,
            "g2": sigmoid_decrease(1.0, 0.3),
        },
        "regulator_activity": "spliced",
        "capture_model": "poisson_capture",
        "capture_rate": 0.5,
        "poisson_observed": True,
        "dropout_rate": 0.02,
        "seed": 7,
    }


def bifurcation_parameters() -> dict[str, object]:
    """Recommended parameter block for a state-anchor bifurcation run."""

    return {
        "n_trunk_cells": 30,
        "n_branch_cells": {"branch_0": 40, "branch_1": 40},
        "trunk_time": 1.8,
        "branch_time": 2.2,
        "dt": 0.05,
        "alpha_source_mode": "state_anchor",
        "production_profile": build_tutorial_profile(),
        "trunk_state": "root",
        "branch_child_states": {"branch_0": "branch_A", "branch_1": "branch_B"},
        "transition_schedule": "sigmoid",
        "transition_midpoint": 0.5,
        "transition_steepness": 10.0,
        "regulator_activity": "spliced",
        "capture_model": "binomial_capture",
        "capture_rate": 0.3,
        "dropout_rate": 0.0,
        "seed": 11,
    }


def run_linear_tutorial() -> dict:
    """Run the linear tutorial simulation."""

    return simulate_linear(build_tutorial_grn(), **linear_parameters())


def run_bifurcation_tutorial() -> dict:
    """Run the bifurcation tutorial simulation."""

    return simulate_bifurcation(build_tutorial_grn(), **bifurcation_parameters())


def _write_if_possible(result: dict, path: Path) -> None:
    try:
        adata = to_anndata(result)
    except ImportError:
        print(f"anndata is not installed; skipped {path.name}")
        return
    adata.write_h5ad(path)


def _print_summary(name: str, result: dict) -> None:
    config = result["uns"]["simulation_config"]
    print(f"[{name}]")
    print(f"  n_cells={result['obs'].shape[0]}")
    print(f"  n_genes={result['var'].shape[0]}")
    print(f"  alpha_source_mode={config['alpha_source_mode']}")
    print(f"  capture_model={config['capture_model']}")
    print(f"  regulator_activity={config['regulator_activity']}")
    if "trunk_state" in config:
        print(f"  trunk_state={config['trunk_state']}")
        print(f"  branch_child_states={config['branch_child_states']}")
        print(f"  transition_schedule={config['transition_schedule']}")


def main() -> None:
    out_dir = Path(__file__).with_name("outputs") / "tutorial"
    out_dir.mkdir(parents=True, exist_ok=True)

    linear_result = run_linear_tutorial()
    bifurcation_result = run_bifurcation_tutorial()

    _write_if_possible(linear_result, out_dir / "tutorial_linear.h5ad")
    _write_if_possible(bifurcation_result, out_dir / "tutorial_bifurcation.h5ad")

    _print_summary("linear", linear_result)
    _print_summary("bifurcation", bifurcation_result)
    print(f"outputs={out_dir}")


if __name__ == "__main__":
    main()
