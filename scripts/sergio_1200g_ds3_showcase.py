#!/usr/bin/env python3
"""Run NVSim on SERGIO's 1200-gene DS3 GRN and draw velocity showcase plots.

This script is intentionally outside ``examples/`` because it depends on a
local SERGIO checkout and the original SERGIO dataset files. It is a reproducible
remote benchmark helper, not a fresh-clone core example.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nvsim.output import to_anndata
from nvsim.plotting import plot_phase_gallery, plot_showcase
from nvsim.production import StateProductionProfile
from nvsim.sergio_io import load_sergio_targets_regs
from nvsim.simulate import simulate_bifurcation

DEFAULT_SERGIO_DATASET_DIR = Path(
    "/mnt/second19T/zhaozelin/simulator/SERGIO/data_sets/De-noised_1200G_9T_300cPerT_6_DS3"
)
DEFAULT_OUTPUT_DIR = ROOT / "examples" / "outputs" / "sergio_1200g_ds3_showcase_20260512"


# ---------------------------------------------------------------------------
# User-selectable model/workflow choices
# ---------------------------------------------------------------------------
# These defaults are chosen to make the run a SERGIO-style state/bin production
# stress test with NVSim's deterministic RNA velocity ODE backend:
#
# - alpha_source_mode="state_anchor": DS3 provides SERGIO bin/state-specific
#   master-regulator production rates in Regs_cID_6.txt. Using state anchors is
#   closer to SERGIO's input semantics than fitting arbitrary continuous alpha
#   programs for each master regulator.
# - trunk_state="bin_0", branch_child_states=("bin_1", "bin_2") by default:
#   this uses the first three SERGIO bins as regulatory anchors. These labels can
#   be changed if you want a different pair of states from the 9-bin DS3 table.
# - transition_schedule="sigmoid": smooth parent-to-child source-alpha changes
#   avoid an artificial hard expression break. Use "step" for hard switching or
#   "linear" for a simpler interpolation sensitivity check.
# - regulator_activity="spliced": NVSim's default mature-RNA/protein proxy. Use
#   "unspliced" for a more SERGIO-compatible dynamic comparison, or "total" for
#   sensitivity analysis.
# - auto_calibrate_half_response="if_missing": SERGIO targets files do not carry
#   explicit half_response values, so NVSim fills missing thresholds from the
#   state-wise mean-expression calibration path before ODE simulation.
# - capture_rate=1.0, poisson_observed=False, dropout_rate=0.0: no technical
#   noise, so the showcase reflects model dynamics rather than count sampling.
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sergio-dataset-dir", type=Path, default=DEFAULT_SERGIO_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trunk-state", default=None, help="Default: first state in Regs_cID_6.txt, usually bin_0")
    parser.add_argument("--branch-0-state", default=None, help="Default: second state in Regs_cID_6.txt, usually bin_1")
    parser.add_argument("--branch-1-state", default=None, help="Default: third state in Regs_cID_6.txt, usually bin_2")
    parser.add_argument("--transition-schedule", choices=["sigmoid", "linear", "step"], default="sigmoid")
    parser.add_argument("--transition-midpoint", type=float, default=0.5)
    parser.add_argument("--transition-steepness", type=float, default=10.0)
    parser.add_argument("--regulator-activity", choices=["spliced", "unspliced", "total"], default="spliced")
    parser.add_argument("--auto-calibrate-half-response", choices=["if_missing", "true", "false"], default="if_missing")
    parser.add_argument("--n-trunk-cells", type=int, default=120)
    parser.add_argument("--n-branch-cells", type=int, default=120)
    parser.add_argument("--trunk-time", type=float, default=2.0)
    parser.add_argument("--branch-time", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--capture-rate", type=float, default=1.0)
    parser.add_argument("--poisson-observed", action="store_true")
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1200)
    parser.add_argument("--expression-layer", choices=["true", "observed"], default="true")
    parser.add_argument("--n-pcs", type=int, default=20)
    parser.add_argument("--n-neighbors", type=int, default=20)
    parser.add_argument("--max-phase-genes", type=int, default=120)
    return parser.parse_args()


def _auto_calibration_value(value: str) -> bool | str:
    if value == "true":
        return True
    if value == "false":
        return False
    return "if_missing"


def _required_file(dataset_dir: Path, name: str) -> Path:
    path = dataset_dir / name
    if not path.exists():
        raise FileNotFoundError(f"missing SERGIO dataset file: {path}")
    return path


def main() -> None:
    args = _parse_args()
    targets = _required_file(args.sergio_dataset_dir, "Interaction_cID_6.txt")
    regs = _required_file(args.sergio_dataset_dir, "Regs_cID_6.txt")

    parsed = load_sergio_targets_regs(targets, regs)
    profile = StateProductionProfile(parsed.master_production)
    states = list(profile.states)
    if len(states) < 3:
        raise ValueError("SERGIO 1200G DS3 state-anchor run requires at least three production states")

    trunk_state = args.trunk_state or states[0]
    branch_child_states = {
        "branch_0": args.branch_0_state or states[1],
        "branch_1": args.branch_1_state or states[2],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = simulate_bifurcation(
        parsed.grn,
        n_trunk_cells=args.n_trunk_cells,
        n_branch_cells={"branch_0": args.n_branch_cells, "branch_1": args.n_branch_cells},
        trunk_time=args.trunk_time,
        branch_time=args.branch_time,
        dt=args.dt,
        master_regulators=parsed.master_regulators,
        production_profile=profile,
        alpha_source_mode="state_anchor",
        trunk_state=trunk_state,
        branch_child_states=branch_child_states,
        transition_schedule=args.transition_schedule,
        transition_midpoint=args.transition_midpoint,
        transition_steepness=args.transition_steepness,
        auto_calibrate_half_response=_auto_calibration_value(args.auto_calibrate_half_response),
        regulator_activity=args.regulator_activity,
        seed=args.seed,
        capture_rate=args.capture_rate,
        poisson_observed=args.poisson_observed,
        dropout_rate=args.dropout_rate,
    )

    h5ad_path = args.output_dir / "sergio_1200g_ds3_state_anchor_bifurcation.h5ad"
    to_anndata(result).write_h5ad(h5ad_path)

    showcase = plot_showcase(
        result,
        output_dir=args.output_dir / "velocity_showcase",
        expression_layer=args.expression_layer,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        min_dist=0.3,
        random_state=args.seed,
    )

    phase_genes = list(result["var"].index[: max(0, args.max_phase_genes)])
    phase_gallery_path = None
    if phase_genes:
        phase_gallery_path = args.output_dir / f"phase_gallery_first_{len(phase_genes)}_genes.png"
        plot_phase_gallery(
            result,
            genes=phase_genes,
            output_path=phase_gallery_path,
            mode="true",
            max_cols=12,
            panel_size=1.15,
        )

    user_choices = {
        "alpha_source_mode": "state_anchor",
        "trunk_state": trunk_state,
        "branch_child_states": branch_child_states,
        "transition_schedule": args.transition_schedule,
        "transition_midpoint": args.transition_midpoint,
        "transition_steepness": args.transition_steepness,
        "regulator_activity": args.regulator_activity,
        "auto_calibrate_half_response": args.auto_calibrate_half_response,
        "capture_rate": args.capture_rate,
        "poisson_observed": args.poisson_observed,
        "dropout_rate": args.dropout_rate,
        "expression_layer_for_showcase": args.expression_layer,
    }
    summary = {
        "dataset_dir": str(args.sergio_dataset_dir),
        "output_dir": str(args.output_dir),
        "h5ad": str(h5ad_path),
        "n_cells": int(result["layers"]["true_spliced"].shape[0]),
        "n_genes": int(result["layers"]["true_spliced"].shape[1]),
        "n_edges": int(len(result["uns"]["true_grn"])),
        "n_master_regulators": int(len(parsed.master_regulators)),
        "available_states": states,
        "user_selectable_choices": user_choices,
        "simulation_config": result["uns"]["simulation_config"],
        "showcase": showcase,
        "extra_plots": {"phase_gallery": str(phase_gallery_path) if phase_gallery_path else None},
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: summary[key] for key in ["output_dir", "h5ad", "n_cells", "n_genes", "n_edges", "n_master_regulators", "user_selectable_choices"]}, indent=2))


if __name__ == "__main__":
    main()
