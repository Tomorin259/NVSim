#!/usr/bin/env python3
"""Run SERGIO 1200G DS3 benchmarks on the unified graph-based NVSim simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nvsim.modes import branching_graph, path_graph
from nvsim.output import to_anndata
from nvsim.plotting import plot_phase_gallery, plot_showcase
from nvsim.production import StateProductionProfile
from nvsim.sergio_io import load_sergio_targets_regs
from nvsim.simulate import simulate

DEFAULT_SERGIO_DATASET_DIR = Path(
    "/mnt/second19T/zhaozelin/simulator/SERGIO/data_sets/De-noised_1200G_9T_300cPerT_6_DS3"
)
DEFAULT_OUTPUT_ROOT = ROOT / "examples" / "outputs"
GRAPH_TEMPLATES = {"path", "branching"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sergio-dataset-dir", type=Path, default=DEFAULT_SERGIO_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--graph-template", choices=sorted(GRAPH_TEMPLATES), default="branching")
    parser.add_argument("--root-state", default="bin_0")
    parser.add_argument("--terminal-state", default=None)
    parser.add_argument("--branch-0-state", default=None)
    parser.add_argument("--branch-1-state", default=None)
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
    parser.add_argument("--n-cells-per-state", type=int, default=300)
    parser.add_argument("--root-time", type=float, default=4.0)
    parser.add_argument("--state-time", type=float, default=6.0)
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


def _select_farthest_states(profile: StateProductionProfile) -> tuple[str, str, dict[str, object]]:
    rates = profile.rates.astype(float)
    states = list(rates.index.astype(str))
    values = rates.to_numpy(dtype=float)
    if len(states) < 2:
        raise ValueError("at least two production states are required")
    best_i = 0
    best_j = 1
    best_distance = -1.0
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            distance = float(np.linalg.norm(values[i] - values[j]))
            if distance > best_distance:
                best_i, best_j, best_distance = i, j, distance
    return states[best_i], states[best_j], {
        "selected_pair": [states[best_i], states[best_j]],
        "selection_method": "farthest_master_production_euclidean_distance",
        "distance": best_distance,
    }


def _select_farthest_child_state(
    profile: StateProductionProfile,
    *,
    parent_state: str,
    allowed_states: list[str],
) -> tuple[str, dict[str, object]]:
    parent = profile.rates.loc[str(parent_state)].to_numpy(dtype=float)
    best_state = None
    best_distance = -1.0
    for state in allowed_states:
        if str(state) == str(parent_state):
            continue
        distance = float(np.linalg.norm(profile.rates.loc[str(state)].to_numpy(dtype=float) - parent))
        if distance > best_distance:
            best_state = str(state)
            best_distance = distance
    if best_state is None:
        raise ValueError("failed to select a terminal state")
    return best_state, {
        "selected_state": best_state,
        "selection_method": "farthest_from_parent_master_production_euclidean_distance",
        "distance": best_distance,
    }


def _select_branch_leaf_states(profile: StateProductionProfile) -> tuple[dict[str, str], dict[str, object]]:
    left, right, meta = _select_farthest_states(profile)
    return {"branch_0": left, "branch_1": right}, meta


def build_path_template(root_state: str, terminal_state: str):
    return path_graph([str(root_state), str(terminal_state)])


def build_branching_template(root_state: str, branch_states: dict[str, str]):
    return branching_graph(str(root_state), [str(branch_states["branch_0"]), str(branch_states["branch_1"])])


def _default_output_dir(graph_template: str) -> Path:
    if graph_template == "path":
        return DEFAULT_OUTPUT_ROOT / "sergio_1200g_ds3_path_graph"
    return DEFAULT_OUTPUT_ROOT / "sergio_1200g_ds3_branching_graph"


def main() -> None:
    args = _parse_args()
    dataset_dir = args.sergio_dataset_dir
    interaction = _required_file(dataset_dir, "Interaction_cID_6.txt")
    regs = _required_file(dataset_dir, "Regs_cID_6.txt")
    inputs = load_sergio_targets_regs(interaction, regs, shared_coop_state=2)
    profile = StateProductionProfile(inputs.master_production)
    states = list(profile.states)

    graph_meta: dict[str, object] = {}
    if args.graph_template == "path":
        terminal_state = args.terminal_state
        if terminal_state is None:
            terminal_state, graph_meta = _select_farthest_child_state(profile, parent_state=args.root_state, allowed_states=states)
        graph = build_path_template(args.root_state, terminal_state)
        n_cells_per_state = {str(args.root_state): args.n_cells_per_state, str(terminal_state): args.n_cells_per_state}
        state_time = {str(args.root_state): args.root_time, str(terminal_state): args.state_time}
    else:
        if args.branch_0_state and args.branch_1_state:
            branch_states = {"branch_0": args.branch_0_state, "branch_1": args.branch_1_state}
            graph_meta = {"selected_pair": [args.branch_0_state, args.branch_1_state], "selection_method": "explicit"}
        else:
            branch_states, graph_meta = _select_branch_leaf_states(profile)
        graph = build_branching_template(args.root_state, branch_states)
        n_cells_per_state = {str(args.root_state): args.n_cells_per_state}
        n_cells_per_state.update({str(v): args.n_cells_per_state for v in branch_states.values()})
        state_time = {state: args.state_time for state in n_cells_per_state}
        state_time[str(args.root_state)] = args.root_time

    result = simulate(
        inputs.grn,
        graph=graph,
        production_profile=profile,
        alpha_source_mode="state_anchor",
        initialization_policy="parent_terminal",
        sampling_policy="state_transient",
        n_cells_per_state=n_cells_per_state,
        root_time=args.root_time,
        state_time=state_time,
        dt=args.dt,
        seed=args.seed,
        capture_rate=args.capture_rate,
        poisson_observed=args.poisson_observed,
        dropout_rate=args.dropout_rate,
        regulator_activity=args.regulator_activity,
        auto_calibrate_half_response=_auto_calibration_value(args.auto_calibrate_half_response),
        transition_schedule=args.transition_schedule,
        transition_midpoint=args.transition_midpoint,
        transition_steepness=args.transition_steepness,
        master_regulators=inputs.master_regulators,
    )

    output_dir = args.output_dir or _default_output_dir(args.graph_template)
    output_dir.mkdir(parents=True, exist_ok=True)
    adata = to_anndata(result)
    h5ad_name = "sergio_1200g_ds3_path_graph.h5ad" if args.graph_template == "path" else "sergio_1200g_ds3_branching_graph.h5ad"
    adata.write_h5ad(output_dir / h5ad_name)

    showcase = plot_showcase(
        result,
        output_dir / "velocity_showcase",
        expression_layer=args.expression_layer,
        basis="umap",
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        random_state=args.seed,
    )
    phase_genes = [str(gene) for gene in adata.var_names[: args.max_phase_genes]]
    plot_phase_gallery(result, genes=phase_genes, output_path=output_dir / "phase_gallery.png")

    summary = {
        "output_dir": str(output_dir),
        "graph_template": args.graph_template,
        "graph_metadata": graph_meta,
        "simulation_config": result["uns"]["simulation_config"],
        "plots": showcase,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
