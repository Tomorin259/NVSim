"""Canonical tutorial examples for the graph-based NVSim API."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nvsim.grn import GRN
from nvsim.modes import StateGraph, branching_graph, path_graph
from nvsim.noise import apply_observation
from nvsim.output import to_anndata
from nvsim.production import StateProductionProfile, linear_increase, sigmoid_decrease
from nvsim.simulate import simulate


def build_tutorial_grn() -> GRN:
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


def linear_graph():
    return path_graph(["early", "late"])


def branching_tutorial_graph():
    return branching_graph("root", ["branch_A", "branch_B"])


def custom_dag_tutorial_graph() -> StateGraph:
    """General DAG-graph example for the canonical graph-based API.

    Path and branching helpers are convenience wrappers. Arbitrary rooted DAGs
    still use the same `simulate(..., graph=...)` entry point. See
    `examples/run_sergio_ds6_dynamic_graph_stepfix.py` for the canonical DS6 benchmark rerun.
    """

    edges = pd.DataFrame(
        {
            "parent_state": ["root", "mid", "mid"],
            "child_state": ["mid", "leaf_A", "leaf_B"],
        }
    )
    return StateGraph(edges, states=("root", "mid", "leaf_A", "leaf_B"))


def linear_parameters() -> dict[str, object]:
    return {
        "n_cells_per_state": {"early": 30, "late": 30},
        "root_time": 1.5,
        "state_time": {"early": 1.5, "late": 1.5},
        "dt": 0.05,
        "alpha_source_mode": "continuous_program",
        "master_programs": {
            "g0": linear_increase(0.2, 1.0),
            "g1": 0.8,
            "g2": sigmoid_decrease(1.0, 0.3),
        },
        "regulator_activity": "spliced",
        "seed": 11,
    }


def branching_parameters() -> dict[str, object]:
    return {
        "graph": branching_tutorial_graph(),
        "production_profile": build_tutorial_profile(),
        "alpha_source_mode": "state_anchor",
        "child_initialization_policy": "parent_terminal",
        "sampling_policy": "state_transient",
        "n_cells_per_state": {"root": 30, "branch_A": 40, "branch_B": 40},
        "root_time": 1.8,
        "state_time": {"root": 1.8, "branch_A": 2.2, "branch_B": 2.2},
        "dt": 0.05,
        "transition_schedule": "sigmoid",
        "transition_midpoint": 0.5,
        "transition_steepness": 10.0,
        "regulator_activity": "spliced",
        "seed": 17,
    }


def run_linear_tutorial() -> dict:
    result = simulate(build_tutorial_grn(), graph=linear_graph(), **linear_parameters())
    return apply_observation(
        result,
        seed=13,
        count_model="poisson",
        cell_capture_mode="constant",
        cell_capture_mean=0.5,
        observation_sample=True,
        dropout_mode="bernoulli",
        dropout_rate=0.02,
    )


def run_branching_tutorial() -> dict:
    params = branching_parameters().copy()
    graph = params.pop("graph")
    result = simulate(build_tutorial_grn(), graph=graph, **params)
    return apply_observation(
        result,
        seed=19,
        count_model="binomial",
        cell_capture_mode="constant",
        cell_capture_mean=0.4,
        observation_sample=True,
        dropout_mode="bernoulli",
        dropout_rate=0.03,
    )


def _write_if_possible(result: dict, path: Path) -> None:
    try:
        adata = to_anndata(result)
        adata.write_h5ad(path)
    except ImportError:
        pass


def _print_summary(label: str, result: dict) -> None:
    print(label)
    print(result["layers"]["true_spliced"].shape)
    print(result["uns"]["simulation_config"]["simulator"])


if __name__ == "__main__":
    out_dir = Path(__file__).with_name("outputs") / "tutorial"
    out_dir.mkdir(parents=True, exist_ok=True)
    linear_result = run_linear_tutorial()
    branching_result = run_branching_tutorial()
    _write_if_possible(linear_result, out_dir / "tutorial_linear_graph.h5ad")
    _write_if_possible(branching_result, out_dir / "tutorial_branching_graph.h5ad")
    _print_summary("linear_graph", linear_result)
    _print_summary("branching_graph", branching_result)
