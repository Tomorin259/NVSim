import numpy as np
import pandas as pd
import pytest

from nvsim.grn import GRN
from nvsim.modes import path_graph
from nvsim.production import linear_increase, sigmoid_decrease
from nvsim.simulate import simulate


def _small_grn():
    genes = ["g0", "g1", "g2", "g3"]
    edges = pd.DataFrame(
        {
            "regulator": ["g0", "g1"],
            "target": ["g2", "g3"],
            "K": [0.7, 0.4],
            "sign": ["activation", "repression"],
            "half_response": [0.5, 0.5],
            "hill_coefficient": [2.0, 2.0],
        }
    )
    return GRN.from_dataframe(edges, genes=genes)


def _small_grn_missing_half_response():
    genes = ["g0", "g1", "g2", "g3"]
    edges = pd.DataFrame(
        {
            "regulator": ["g0", "g1"],
            "target": ["g2", "g3"],
            "K": [0.7, 0.4],
            "sign": ["activation", "repression"],
            "hill_coefficient": [2.0, 2.0],
        }
    )
    return GRN.from_dataframe(edges, genes=genes)


def _chain_kwargs():
    return {
        "graph": path_graph(["early", "late"]),
        "alpha_source_mode": "continuous_program",
        "master_programs": {
            "g0": linear_increase(0.2, 1.0),
            "g1": sigmoid_decrease(1.0, 0.3),
        },
        "n_cells_per_state": {"early": 6, "late": 6},
        "root_time": 1.0,
        "state_time": {"early": 1.0, "late": 1.0},
        "dt": 0.05,
        "seed": 31,
        "poisson_observed": False,
    }


def test_true_velocity_matches_formula_and_states_are_nonnegative():
    grn = _small_grn()
    result = simulate(grn, **_chain_kwargs())
    layers = result["layers"]
    beta = result["uns"]["kinetic_params"]["beta"].to_numpy()
    gamma = result["uns"]["kinetic_params"]["gamma"].to_numpy()
    expected = layers["true_unspliced"] * beta - layers["true_spliced"] * gamma
    expected_u = layers["true_alpha"] - layers["true_unspliced"] * beta

    assert np.allclose(layers["true_velocity"], expected)
    assert np.allclose(layers["true_velocity_u"], expected_u)
    assert np.all(layers["true_alpha"] >= 0.0)
    assert np.all(layers["true_unspliced"] >= 0.0)
    assert np.all(layers["true_spliced"] >= 0.0)


def test_output_dimensions_are_correct():
    grn = _small_grn()
    result = simulate(grn, **_chain_kwargs())

    for layer in result["layers"].values():
        assert layer.shape == (12, 4)
    assert result["obs"].shape[0] == 12
    assert result["var"].shape[0] == 4
    assert set(["pseudotime", "branch", "state"]).issubset(result["obs"].columns)


def test_var_metadata_distinguishes_gene_role_from_gene_class():
    grn = _small_grn()
    result = simulate(grn, **_chain_kwargs())
    var = result["var"]

    assert set(["gene_role", "gene_class", "gene_level", "true_beta", "true_gamma"]).issubset(var.columns)
    assert set(var["gene_role"]) == {"master_regulator", "target"}
    assert set(var["gene_class"]) == {"unassigned"}


def test_seed_is_reproducible():
    grn = _small_grn()
    result1 = simulate(grn, **_chain_kwargs())
    result2 = simulate(grn, **_chain_kwargs())

    assert np.allclose(result1["layers"]["true_spliced"], result2["layers"]["true_spliced"])
    assert np.allclose(result1["layers"]["true_unspliced"], result2["layers"]["true_unspliced"])
    assert result1["obs"].equals(result2["obs"])


def test_sampling_without_replacement_is_default():
    grn = _small_grn()
    with pytest.raises(ValueError, match="n_cells exceeds available timepoints"):
        simulate(
            grn,
            graph=path_graph(["s0"]),
            alpha_source_mode="continuous_program",
            master_programs={"g0": 0.5, "g1": 0.5},
            n_cells_per_state={"s0": 50},
            root_time=0.2,
            state_time={"s0": 0.2},
            dt=0.1,
            seed=11,
            poisson_observed=False,
        )


def test_sampling_replacement_can_be_enabled_and_is_recorded():
    grn = _small_grn()
    result = simulate(
        grn,
        graph=path_graph(["s0"]),
        alpha_source_mode="continuous_program",
        master_programs={"g0": 0.5, "g1": 0.5},
        n_cells_per_state={"s0": 50},
        root_time=0.2,
        state_time={"s0": 0.2},
        dt=0.1,
        seed=11,
        poisson_observed=False,
        allow_snapshot_replacement=True,
    )

    assert result["uns"]["simulation_config"]["sampling_replace"] is True


def test_explicit_initial_state_is_recorded_for_roots():
    grn = _small_grn()
    kwargs = _chain_kwargs()
    kwargs["u0"] = np.full(4, 0.2)
    kwargs["s0"] = np.full(4, 0.3)
    result = simulate(grn, **kwargs)

    assert result["uns"]["state_initialization"]["early"]["source"] == "explicit_initial_state"


def test_auto_half_response_calibration_works_for_continuous_program():
    grn = _small_grn_missing_half_response()
    result = simulate(grn, **_chain_kwargs(), half_response_calibration="auto")
    assert result["uns"]["simulation_config"]["half_response_calibration"] == "auto"
    assert result["uns"]["grn_calibration"]["actual_calibration"] == "topology_propagation"


def test_invalid_regulator_activity_is_rejected():
    grn = _small_grn()
    with pytest.raises(ValueError, match="regulator_activity"):
        simulate(grn, **_chain_kwargs(), regulator_activity="bad_mode")


def test_removed_linear_alias_is_rejected():
    grn = _small_grn()
    with pytest.raises(ValueError, match="simulator must be"):
        simulate(grn, simulator="linear", graph=path_graph(["early", "late"]))


def test_removed_random_small_behavior_is_gone_for_roots():
    grn = _small_grn()
    result = simulate(grn, **_chain_kwargs())
    root_init = result["uns"]["state_initialization"]["early"]
    assert root_init["source"] == "state_steady_state"


def test_sergio_kinetics_mode_rejects_explicit_beta_gamma():
    grn = _small_grn()
    with pytest.raises(ValueError, match="kinetics_mode='sergio'"):
        simulate(grn, **_chain_kwargs(), kinetics_mode="sergio", beta=np.full(4, 0.8))
