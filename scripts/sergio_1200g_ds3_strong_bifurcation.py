#!/usr/bin/env python3
"""Run SERGIO 1200G DS3 NVSim benchmarks with a unified simulator entry.

This script supports multiple trajectory simulators behind one CLI:
- linear
- bifurcation (alias: branch)

It depends on external SERGIO dataset files and is intended as a remote
benchmark/showcase helper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nvsim.output import to_anndata
from nvsim.plotting import plot_phase_gallery, plot_showcase
from nvsim.production import StateProductionProfile
from nvsim.sergio_io import load_sergio_targets_regs
from nvsim.simulate import simulate_bifurcation, simulate_linear

DEFAULT_SERGIO_DATASET_DIR = Path(
    "/mnt/second19T/zhaozelin/simulator/SERGIO/data_sets/De-noised_1200G_9T_300cPerT_6_DS3"
)
DEFAULT_OUTPUT_ROOT = ROOT / "examples" / "outputs"

SIMULATOR_ALIASES = {
    "linear": "linear",
    "branch": "bifurcation",
    "bifurcation": "bifurcation",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sergio-dataset-dir", type=Path, default=DEFAULT_SERGIO_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--simulator",
        choices=sorted(SIMULATOR_ALIASES),
        default="branch",
        help="Trajectory simulator: linear or branch/bifurcation.",
    )

    parser.add_argument("--trunk-state", default="bin_0")
    parser.add_argument("--transition-schedule", choices=["sigmoid", "linear", "step"], default="sigmoid")
    parser.add_argument("--transition-midpoint", type=float, default=0.25)
    parser.add_argument("--transition-steepness", type=float, default=20.0)
    parser.add_argument("--regulator-activity", choices=["spliced", "unspliced", "total"], default="unspliced")
    parser.add_argument("--auto-calibrate-half-response", choices=["if_missing", "true", "false"], default="if_missing")

    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--capture-rate", type=float, default=1.0)
    parser.add_argument("--poisson-observed", action="store_true")
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1200)

    parser.add_argument("--expression-layer", choices=["true", "observed"], default="true")
    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--max-phase-genes", type=int, default=120)

    # Linear-specific options.
    parser.add_argument("--n-cells", type=int, default=780)
    parser.add_argument("--time-end", type=float, default=10.0)
    parser.add_argument("--child-state", default=None)
    parser.add_argument("--no-auto-select-child-state", action="store_true")

    # Bifurcation-specific options.
    parser.add_argument("--branch-0-state", default=None)
    parser.add_argument("--branch-1-state", default=None)
    parser.add_argument("--no-auto-select-branch-states", action="store_true")
    parser.add_argument("--n-trunk-cells", type=int, default=180)
    parser.add_argument("--n-branch-cells", type=int, default=300)
    parser.add_argument("--trunk-time", type=float, default=4.0)
    parser.add_argument("--branch-time", type=float, default=6.0)

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


def _resolve_simulator(simulator: str) -> str:
    return SIMULATOR_ALIASES[str(simulator).strip().lower()]


def _default_output_dir(simulator: str) -> Path:
    if simulator == "linear":
        return DEFAULT_OUTPUT_ROOT / "sergio_1200g_ds3_linear_state_anchor"
    return DEFAULT_OUTPUT_ROOT / "sergio_1200g_ds3_strong_bifurcation_unspliced_npcs30"


def _select_farthest_states(profile: StateProductionProfile) -> tuple[str, str, dict[str, object]]:
    rates = profile.rates.astype(float)
    states = list(rates.index.astype(str))
    values = rates.to_numpy(dtype=float)
    if len(states) < 2:
        raise ValueError("at least two production states are required for branch selection")

    best = (states[0], states[1], -1.0)
    pairwise: list[dict[str, object]] = []
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            dist = float(np.linalg.norm(values[i] - values[j]))
            pairwise.append({"state_0": states[i], "state_1": states[j], "distance": dist})
            if dist > best[2]:
                best = (states[i], states[j], dist)

    return best[0], best[1], {
        "selection_method": "farthest_master_production_euclidean_distance",
        "selected_pair": [best[0], best[1]],
        "selected_distance": best[2],
        "pairwise_distances": sorted(pairwise, key=lambda item: item["distance"], reverse=True),
    }


def _select_farthest_child_state(
    profile: StateProductionProfile,
    *,
    parent_state: str,
    allowed_states: list[str],
) -> tuple[str, dict[str, object]]:
    rates = profile.rates.astype(float)
    if parent_state not in allowed_states:
        raise ValueError(f"unknown parent_state {parent_state!r}; available states: {allowed_states}")

    parent = rates.loc[parent_state].to_numpy(dtype=float)
    ranked: list[dict[str, object]] = []
    for state in allowed_states:
        if state == parent_state:
            continue
        dist = float(np.linalg.norm(rates.loc[state].to_numpy(dtype=float) - parent))
        ranked.append({"state": state, "distance_from_parent": dist})
    if not ranked:
        raise ValueError("at least one child state different from parent_state is required")
    ranked = sorted(ranked, key=lambda item: item["distance_from_parent"], reverse=True)
    return ranked[0]["state"], {
        "selection_method": "farthest_from_parent_master_production_euclidean_distance",
        "parent_state": parent_state,
        "selected_state": ranked[0]["state"],
        "selected_distance": ranked[0]["distance_from_parent"],
        "candidate_distances": ranked,
    }


def _resolve_branch_states(args: argparse.Namespace, states: list[str], profile: StateProductionProfile) -> tuple[dict[str, str], dict[str, object]]:
    selection: dict[str, object] = {
        "selection_method": "explicit_or_default",
        "selected_pair": [args.branch_0_state, args.branch_1_state],
    }
    if args.no_auto_select_branch_states:
        branch_0_state = args.branch_0_state or states[1]
        branch_1_state = args.branch_1_state or states[2]
    elif args.branch_0_state is None and args.branch_1_state is None:
        branch_0_state, branch_1_state, selection = _select_farthest_states(profile)
    elif args.branch_0_state is not None and args.branch_1_state is not None:
        branch_0_state, branch_1_state = args.branch_0_state, args.branch_1_state
    else:
        raise ValueError("provide both --branch-0-state and --branch-1-state, or neither for auto-selection")

    if branch_0_state not in states or branch_1_state not in states:
        raise ValueError(f"unknown branch child states; available states: {states}")
    return {"branch_0": branch_0_state, "branch_1": branch_1_state}, selection


def _resolve_linear_child_state(args: argparse.Namespace, states: list[str], profile: StateProductionProfile) -> tuple[str, dict[str, object]]:
    if args.no_auto_select_child_state:
        if args.child_state is None:
            fallback = next((state for state in states if state != args.trunk_state), None)
            if fallback is None:
                raise ValueError("unable to resolve child state; dataset has only one state")
            child_state = fallback
            selection = {
                "selection_method": "default_first_non_parent_state",
                "parent_state": args.trunk_state,
                "selected_state": child_state,
            }
        else:
            child_state = args.child_state
            selection = {
                "selection_method": "explicit",
                "parent_state": args.trunk_state,
                "selected_state": child_state,
            }
    elif args.child_state is None:
        child_state, selection = _select_farthest_child_state(profile, parent_state=args.trunk_state, allowed_states=states)
    else:
        child_state = args.child_state
        selection = {
            "selection_method": "explicit",
            "parent_state": args.trunk_state,
            "selected_state": child_state,
        }

    if child_state not in states:
        raise ValueError(f"unknown child_state {child_state!r}; available states: {states}")
    if child_state == args.trunk_state:
        raise ValueError("child_state must differ from trunk_state for linear state-anchor transition")
    return child_state, selection


def main() -> None:
    args = _parse_args()
    simulator = _resolve_simulator(args.simulator)

    targets = _required_file(args.sergio_dataset_dir, "Interaction_cID_6.txt")
    regs = _required_file(args.sergio_dataset_dir, "Regs_cID_6.txt")

    parsed = load_sergio_targets_regs(targets, regs)
    profile = StateProductionProfile(parsed.master_production)
    states = list(profile.states)
    if args.trunk_state not in states:
        raise ValueError(f"unknown trunk_state {args.trunk_state!r}; available states: {states}")

    output_dir = args.output_dir or _default_output_dir(simulator)
    output_dir.mkdir(parents=True, exist_ok=True)

    auto_calibration = _auto_calibration_value(args.auto_calibrate_half_response)

    trajectory_config: dict[str, object]
    user_choices: dict[str, object]
    if simulator == "bifurcation":
        branch_child_states, branch_selection = _resolve_branch_states(args, states, profile)
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
            trunk_state=args.trunk_state,
            branch_child_states=branch_child_states,
            transition_schedule=args.transition_schedule,
            transition_midpoint=args.transition_midpoint,
            transition_steepness=args.transition_steepness,
            auto_calibrate_half_response=auto_calibration,
            regulator_activity=args.regulator_activity,
            seed=args.seed,
            capture_rate=args.capture_rate,
            poisson_observed=args.poisson_observed,
            dropout_rate=args.dropout_rate,
        )
        trajectory_config = {
            "simulator": "bifurcation",
            "trunk_state": args.trunk_state,
            "branch_child_states": branch_child_states,
            "branch_state_selection": branch_selection,
            "n_trunk_cells": args.n_trunk_cells,
            "n_branch_cells": args.n_branch_cells,
            "trunk_time": args.trunk_time,
            "branch_time": args.branch_time,
        }
    else:
        child_state, child_selection = _resolve_linear_child_state(args, states, profile)
        result = simulate_linear(
            parsed.grn,
            n_cells=args.n_cells,
            time_end=args.time_end,
            dt=args.dt,
            master_regulators=parsed.master_regulators,
            production_profile=profile,
            alpha_source_mode="state_anchor",
            parent_state=args.trunk_state,
            child_state=child_state,
            transition_schedule=args.transition_schedule,
            transition_midpoint=args.transition_midpoint,
            transition_steepness=args.transition_steepness,
            auto_calibrate_half_response=auto_calibration,
            regulator_activity=args.regulator_activity,
            seed=args.seed,
            capture_rate=args.capture_rate,
            poisson_observed=args.poisson_observed,
            dropout_rate=args.dropout_rate,
        )
        trajectory_config = {
            "simulator": "linear",
            "parent_state": args.trunk_state,
            "child_state": child_state,
            "child_state_selection": child_selection,
            "n_cells": args.n_cells,
            "time_end": args.time_end,
        }

    h5ad_name = "sergio_1200g_ds3_linear.h5ad" if simulator == "linear" else "sergio_1200g_ds3_strong_bifurcation.h5ad"
    h5ad_path = output_dir / h5ad_name
    to_anndata(result).write_h5ad(h5ad_path)

    showcase = plot_showcase(
        result,
        output_dir=output_dir / "velocity_showcase",
        expression_layer=args.expression_layer,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        min_dist=0.3,
        random_state=args.seed,
    )

    phase_genes = list(result["var"].index[: max(0, args.max_phase_genes)])
    phase_gallery_path = None
    if phase_genes:
        phase_gallery_path = output_dir / f"phase_gallery_first_{len(phase_genes)}_genes.png"
        plot_phase_gallery(
            result,
            genes=phase_genes,
            output_path=phase_gallery_path,
            mode="true",
            max_cols=12,
            panel_size=1.15,
        )

    user_choices = {
        "simulator": args.simulator,
        "resolved_simulator": simulator,
        "alpha_source_mode": "state_anchor",
        "trunk_state": args.trunk_state,
        "transition_schedule": args.transition_schedule,
        "transition_midpoint": args.transition_midpoint,
        "transition_steepness": args.transition_steepness,
        "regulator_activity": args.regulator_activity,
        "auto_calibrate_half_response": args.auto_calibrate_half_response,
        "capture_rate": args.capture_rate,
        "poisson_observed": args.poisson_observed,
        "dropout_rate": args.dropout_rate,
        "expression_layer_for_showcase": args.expression_layer,
        "n_pcs": args.n_pcs,
        "n_neighbors": args.n_neighbors,
        "dt": args.dt,
        **trajectory_config,
    }

    summary = {
        "dataset_dir": str(args.sergio_dataset_dir),
        "output_dir": str(output_dir),
        "h5ad": str(h5ad_path),
        "simulator": simulator,
        "n_cells": int(result["layers"]["true_spliced"].shape[0]),
        "n_genes": int(result["layers"]["true_spliced"].shape[1]),
        "n_edges": int(len(result["uns"]["true_grn"])),
        "n_master_regulators": int(len(parsed.master_regulators)),
        "available_states": states,
        "trajectory_config": trajectory_config,
        "user_selectable_choices": user_choices,
        "simulation_config": result["uns"]["simulation_config"],
        "showcase": showcase,
        "extra_plots": {"phase_gallery": str(phase_gallery_path) if phase_gallery_path else None},
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in ["output_dir", "h5ad", "simulator", "n_cells", "n_genes", "n_edges", "n_master_regulators", "trajectory_config"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
