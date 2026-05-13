import numpy as np
import pandas as pd
import pytest

from nvsim.grn import GRN
from nvsim.modes import StateGraph, branching_graph, path_graph
from nvsim.production import StateProductionProfile
from nvsim.simulate import simulate


def _mode_grn() -> GRN:
    edges = pd.DataFrame(
        {
            "regulator": ["g0", "g1"],
            "target": ["g2", "g3"],
            "K": [0.8, 0.6],
            "sign": ["activation", "repression"],
            "half_response": [0.5, 0.5],
            "hill_coefficient": [2.0, 2.0],
        }
    )
    return GRN.from_dataframe(edges, genes=["g0", "g1", "g2", "g3"], master_regulators=["g0", "g1"])


def _mode_profile() -> StateProductionProfile:
    return StateProductionProfile(
        pd.DataFrame(
            {
                "g0": [0.6, 1.2, 0.2],
                "g1": [0.5, 0.3, 1.1],
            },
            index=["root", "branch_a", "branch_b"],
        )
    )


def _mode_graph() -> StateGraph:
    return branching_graph("root", ["branch_a", "branch_b"])


def test_sergio_differentiation_mode_resolves_expected_defaults():
    result = simulate(
        _mode_grn(),
        simulation_mode="sergio_differentiation",
        graph=_mode_graph(),
        production_profile=_mode_profile(),
        n_cells_per_state=5,
        root_time=0.5,
        state_time=0.4,
        dt=0.1,
        seed=9,
        poisson_observed=False,
    )
    config = result["uns"]["simulation_config"]

    assert config["simulation_mode"] == "sergio_differentiation"
    assert config["simulator"] == "graph"
    assert config["alpha_source_mode"] == "state_anchor"
    assert config["initialization_policy"] == "parent_steady_state"
    assert config["sampling_policy"] == "state_transient"
    assert config["transition_schedule"] == "step"
    assert config["regulator_activity"] == "unspliced"
    assert set(result["obs"]["state"].unique()) == {"root", "branch_a", "branch_b"}
    assert set(["state", "parent_state", "edge_id", "state_depth"]).issubset(result["obs"].columns)


def test_sergio_differentiation_children_start_from_parent_steady_state_by_default():
    result = simulate(
        _mode_grn(),
        simulation_mode="sergio_differentiation",
        graph=_mode_graph(),
        production_profile=_mode_profile(),
        n_cells_per_state=4,
        root_time=0.5,
        state_time=0.4,
        dt=0.1,
        seed=4,
        poisson_observed=False,
    )

    steady = result["uns"]["state_steady_states"]["root"]["u"]
    init = result["uns"]["state_initialization"]["branch_a"]
    assert init["source"] == "parent_state_steady_state"
    assert init["parent_state"] == "root"
    assert np.allclose(init["u0"], steady)


def test_simulator_rejects_removed_special_cases():
    with pytest.raises(ValueError, match="simulator must be"):
        simulate(_mode_grn(), simulator="linear", graph=_mode_graph(), production_profile=_mode_profile())


def test_state_graph_rejects_cycles():
    with pytest.raises(ValueError, match="acyclic"):
        StateGraph(
            pd.DataFrame(
                {
                    "parent_state": ["a", "b"],
                    "child_state": ["b", "a"],
                }
            )
        )


def test_state_graph_rejects_multiple_parents():
    with pytest.raises(ValueError, match="at most one parent"):
        StateGraph(
            pd.DataFrame(
                {
                    "parent_state": ["root_0", "root_1"],
                    "child_state": ["child", "child"],
                }
            )
        )


def test_path_graph_helper_builds_chain_order():
    graph = path_graph(["s0", "s1", "s2"])
    assert graph.topological_order() == ("s0", "s1", "s2")
    assert graph.parent_of("s2") == "s1"
