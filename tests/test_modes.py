import numpy as np
import pandas as pd
import pytest

from nvsim.grn import GRN
from nvsim.modes import DifferentiationGraph
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


def _mode_graph() -> DifferentiationGraph:
    return DifferentiationGraph(
        pd.DataFrame(
            {
                "parent_state": ["root", "root"],
                "child_state": ["branch_a", "branch_b"],
            }
        )
    )


def test_sergio_differentiation_mode_resolves_expected_defaults():
    result = simulate(
        _mode_grn(),
        simulation_mode="sergio_differentiation",
        differentiation_graph=_mode_graph(),
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
    assert config["simulator"] == "differentiation_graph"
    assert config["alpha_source_mode"] == "state_anchor"
    assert config["initialization_policy"] == "state_anchor_steady_state"
    assert config["sampling_policy"] == "sergio_transient"
    assert config["transition_schedule"] == "step"
    assert config["regulator_activity"] == "unspliced"
    assert set(result["obs"]["state"].unique()) == {"root", "branch_a", "branch_b"}
    assert set(["state", "parent_state", "edge_id", "state_depth"]).issubset(result["obs"].columns)


def test_sergio_differentiation_children_start_from_parent_steady_state_by_default():
    result = simulate(
        _mode_grn(),
        simulation_mode="sergio_differentiation",
        differentiation_graph=_mode_graph(),
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


def test_sergio_differentiation_mode_rejects_incompatible_simulator_override():
    with pytest.raises(ValueError, match="incompatible with simulator"):
        simulate(
            _mode_grn(),
            simulator="linear",
            simulation_mode="sergio_differentiation",
            differentiation_graph=_mode_graph(),
            production_profile=_mode_profile(),
        )


def test_differentiation_graph_rejects_cycles():
    with pytest.raises(ValueError, match="acyclic"):
        DifferentiationGraph(
            pd.DataFrame(
                {
                    "parent_state": ["a", "b"],
                    "child_state": ["b", "a"],
                }
            )
        )


def test_differentiation_graph_rejects_multiple_parents():
    with pytest.raises(ValueError, match="at most one parent"):
        DifferentiationGraph(
            pd.DataFrame(
                {
                    "parent_state": ["root_0", "root_1"],
                    "child_state": ["child", "child"],
                }
            )
        )
