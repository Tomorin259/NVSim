import numpy as np
import pandas as pd
import pytest

from nvsim.grn import GRN
from nvsim.modes import branching_graph
from nvsim.noise import apply_observation
from nvsim.production import StateProductionProfile
from nvsim.simulate import simulate


def _graph_grn() -> GRN:
    edges = pd.DataFrame(
        {
            "regulator": ["g0", "g1"],
            "target": ["g2", "g3"],
            "K": [0.8, 0.7],
            "sign": ["activation", "repression"],
            "half_response": [0.5, 0.5],
            "hill_coefficient": [2.0, 2.0],
        }
    )
    return GRN.from_dataframe(edges, genes=["g0", "g1", "g2", "g3"], master_regulators=["g0", "g1"])


def _graph_profile() -> StateProductionProfile:
    return StateProductionProfile(
        pd.DataFrame(
            {
                "g0": [0.5, 1.3, 0.2],
                "g1": [0.4, 0.1, 1.1],
            },
            index=["root", "branch_0", "branch_1"],
        )
    )


def _branching_kwargs():
    return {
        "graph": branching_graph("root", ["branch_0", "branch_1"]),
        "production_profile": _graph_profile(),
        "alpha_source_mode": "state_anchor",
        "n_cells_per_state": {"root": 8, "branch_0": 9, "branch_1": 10},
        "root_time": 1.0,
        "state_time": {"root": 1.0, "branch_0": 1.2, "branch_1": 1.2},
        "dt": 0.05,
        "seed": 5,
    }


def test_branching_output_shapes_and_obs_alignment():
    result = simulate(_graph_grn(), **_branching_kwargs())

    assert result["layers"]["true_spliced"].shape == (27, 4)
    assert list(result["obs"]["state"].unique()) == ["root", "branch_0", "branch_1"]
    assert set(result["obs"]["parent_state"].fillna("<root>").unique()) == {"<root>", "root"}


def test_branching_true_velocity_matches_formula():
    result = simulate(_graph_grn(), **_branching_kwargs())
    beta = result["uns"]["kinetic_params"]["beta"].to_numpy()
    gamma = result["uns"]["kinetic_params"]["gamma"].to_numpy()
    expected = result["layers"]["true_unspliced"] * beta - result["layers"]["true_spliced"] * gamma
    assert np.allclose(result["layers"]["true_velocity"], expected)


def test_branching_seed_is_reproducible():
    result1 = simulate(_graph_grn(), **_branching_kwargs())
    result2 = simulate(_graph_grn(), **_branching_kwargs())

    assert np.allclose(result1["layers"]["true_spliced"], result2["layers"]["true_spliced"])
    assert result1["obs"].equals(result2["obs"])


def test_branching_parent_terminal_is_default_child_initialization():
    result = simulate(_graph_grn(), **_branching_kwargs())
    assert result["uns"]["state_initialization"]["branch_0"]["source"] == "parent_terminal_state"
    assert result["uns"]["state_initialization"]["branch_1"]["source"] == "parent_terminal_state"


def test_branching_can_use_parent_steady_state_initialization():
    result = simulate(_graph_grn(), **_branching_kwargs(), child_initialization_policy="parent_steady_state")
    assert result["uns"]["state_initialization"]["branch_0"]["source"] == "parent_state_steady_state"


def test_branching_state_anchor_requires_known_states():
    profile = StateProductionProfile(pd.DataFrame({"g0": [1.0], "g1": [1.0]}, index=["other"]))
    kwargs = _branching_kwargs()
    kwargs["production_profile"] = profile
    with pytest.raises(ValueError):
        simulate(_graph_grn(), **kwargs)


def test_branching_can_auto_calibrate_missing_half_response():
    grn = GRN.from_dataframe(
        pd.DataFrame(
            {
                "regulator": ["g0", "g1"],
                "target": ["g2", "g3"],
                "K": [0.8, 0.7],
                "sign": ["activation", "repression"],
                "hill_coefficient": [2.0, 2.0],
            }
        ),
        genes=["g0", "g1", "g2", "g3"],
        master_regulators=["g0", "g1"],
    )
    result = simulate(grn, **_branching_kwargs(), half_response_calibration="auto")
    assert result["uns"]["simulation_config"]["half_response_calibration"] == "auto"


def test_branching_regulator_activity_modes_change_dynamics():
    kwargs = _branching_kwargs()
    spliced = simulate(_graph_grn(), regulator_activity="spliced", **kwargs)
    unspliced = simulate(_graph_grn(), regulator_activity="unspliced", **kwargs)
    total = simulate(_graph_grn(), regulator_activity="total", **kwargs)

    assert not np.allclose(spliced["layers"]["true_spliced"], unspliced["layers"]["true_spliced"])
    assert not np.allclose(unspliced["layers"]["true_spliced"], total["layers"]["true_spliced"])


def test_branching_output_contains_standardized_metadata():
    result = simulate(_graph_grn(), **_branching_kwargs())
    config = result["uns"]["simulation_config"]

    assert config["simulator"] == "graph"
    assert config["alpha_source_mode"] == "state_anchor"
    assert config["production_profile"] is True
    assert config["root_states"] == ["root"]
    assert len(config["graph_edges"]) == 2


def test_branching_observation_metadata_is_recorded():
    clean = simulate(_graph_grn(), **_branching_kwargs())
    result = apply_observation(clean, count_model="binomial", cell_capture_mean=0.5)
    assert result["uns"]["observation_config"]["count_model"] == "binomial"
    assert result["uns"]["observation_config"]["cell_capture_mean"] == 0.5


def test_sergio_differentiation_defaults_to_sergio_kinetics():
    result = simulate(
        _graph_grn(),
        **_branching_kwargs(),
        simulation_mode="sergio_differentiation",
    )
    beta = result["uns"]["kinetic_params"]["beta"].to_numpy()
    gamma = result["uns"]["kinetic_params"]["gamma"].to_numpy()

    assert result["uns"]["simulation_config"]["kinetics_mode"] == "sergio"
    assert result["uns"]["kinetic_params"]["kinetics_mode"] == "sergio"
    assert result["uns"]["kinetic_params"]["sergio_decay"] == pytest.approx(0.8)
    assert result["uns"]["kinetic_params"]["sergio_splice_ratio"] == pytest.approx(4.0)
    assert np.allclose(beta, 0.8)
    assert np.allclose(gamma, 0.2)


def test_branching_sampling_without_replacement_is_default():
    with pytest.raises(ValueError, match="n_cells exceeds available timepoints"):
        simulate(
            _graph_grn(),
            graph=branching_graph("root", ["branch_0", "branch_1"]),
            production_profile=_graph_profile(),
            alpha_source_mode="state_anchor",
            n_cells_per_state={"root": 50, "branch_0": 50, "branch_1": 50},
            root_time=0.2,
            state_time={"root": 0.2, "branch_0": 0.2, "branch_1": 0.2},
            dt=0.1,
            seed=11,
        )


def test_branching_sampling_replacement_can_be_enabled_and_is_recorded():
    result = simulate(
        _graph_grn(),
        graph=branching_graph("root", ["branch_0", "branch_1"]),
        production_profile=_graph_profile(),
        alpha_source_mode="state_anchor",
        n_cells_per_state={"root": 50, "branch_0": 50, "branch_1": 50},
        root_time=0.2,
        state_time={"root": 0.2, "branch_0": 0.2, "branch_1": 0.2},
        dt=0.1,
        seed=11,
        allow_snapshot_replacement=True,
    )
    assert result["uns"]["simulation_config"]["sampling_replace"] is True
