import numpy as np
import pandas as pd
import pytest

from nvsim.grn import GRN
from nvsim.production import StateProductionProfile
from nvsim.production import linear_decrease, linear_increase
from nvsim.simulate import simulate_bifurcation


def _small_grn():
    genes = ["g0", "g1", "g2", "g3"]
    edges = pd.DataFrame(
        {
            "regulator": ["g0", "g1"],
            "target": ["g2", "g3"],
            "K": [0.7, 0.4],
            "sign": ["activation", "repression"],
            "threshold": [0.5, 0.5],
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


def _result(seed=17):
    return simulate_bifurcation(
        _small_grn(),
        n_trunk_cells=8,
        n_branch_cells={"branch_0": 9, "branch_1": 10},
        trunk_time=1.0,
        branch_time=1.2,
        dt=0.04,
        master_programs={"g0": linear_increase(0.2, 0.7), "g1": 0.5},
        branch_master_programs={
            "branch_0": {"g0": linear_increase(0.7, 1.4)},
            "branch_1": {"g0": linear_decrease(0.7, 0.1)},
        },
        seed=seed,
        poisson_observed=False,
    )


def test_branch_initial_states_inherit_trunk_terminal_state():
    result = _result()
    inheritance = result["uns"]["branch_inheritance"]
    assert np.allclose(inheritance["branch_0_initial_u"], inheritance["trunk_terminal_u"])
    assert np.allclose(inheritance["branch_0_initial_s"], inheritance["trunk_terminal_s"])
    assert np.allclose(inheritance["branch_1_initial_u"], inheritance["trunk_terminal_u"])
    assert np.allclose(inheritance["branch_1_initial_s"], inheritance["trunk_terminal_s"])

    segments = result["uns"]["segment_time_courses"]
    assert np.allclose(segments["branch_0"]["u"][0], segments["trunk"]["u"][-1])
    assert np.allclose(segments["branch_1"]["s"][0], segments["trunk"]["s"][-1])


def test_branches_evolve_independently_after_branch_point():
    result = _result()
    segments = result["uns"]["segment_time_courses"]
    assert not np.allclose(segments["branch_0"]["s"][-1], segments["branch_1"]["s"][-1])
    assert not np.allclose(segments["branch_0"]["alpha"][-1], segments["branch_1"]["alpha"][-1])


def test_bifurcation_output_shapes_and_obs_alignment():
    result = _result()
    n_cells = 27
    n_genes = 4
    for layer in result["layers"].values():
        assert layer.shape == (n_cells, n_genes)
    assert list(result["obs"].columns) == ["pseudotime", "local_time", "branch", "segment", "time_index"]
    assert result["obs"].shape[0] == n_cells
    assert list(result["obs"]["branch"].unique()) == ["trunk", "branch_0", "branch_1"]
    assert (result["obs"].loc[result["obs"]["branch"] == "trunk", "pseudotime"] <= 1.0).all()
    assert (result["obs"].loc[result["obs"]["branch"] != "trunk", "local_time"] >= 0.0).all()


def test_bifurcation_true_velocity_matches_formula():
    result = _result()
    layers = result["layers"]
    beta = result["uns"]["kinetic_params"]["beta"].to_numpy()
    gamma = result["uns"]["kinetic_params"]["gamma"].to_numpy()
    expected = layers["true_unspliced"] * beta - layers["true_spliced"] * gamma
    expected_u = layers["true_alpha"] - layers["true_unspliced"] * beta
    assert np.allclose(layers["true_velocity"], expected)
    assert np.allclose(layers["true_velocity_u"], expected_u)


def test_bifurcation_seed_is_reproducible():
    result1 = _result(seed=31)
    result2 = _result(seed=31)
    assert np.array_equal(result1["obs"]["time_index"].to_numpy(), result2["obs"]["time_index"].to_numpy())
    assert np.allclose(result1["layers"]["true_spliced"], result2["layers"]["true_spliced"])
    assert np.array_equal(result1["layers"]["spliced"], result2["layers"]["spliced"])


def test_bifurcation_can_use_state_production_profile():
    production = StateProductionProfile(
        pd.DataFrame(
            {
                "g0": [0.4, 1.2, 0.1],
                "g1": [0.5, 0.2, 1.3],
            },
            index=["trunk_state", "branch_0_state", "branch_1_state"],
        )
    )

    result = simulate_bifurcation(
        _small_grn(),
        n_trunk_cells=6,
        n_branch_cells={"branch_0": 6, "branch_1": 6},
        trunk_time=0.8,
        branch_time=0.8,
        dt=0.04,
        production_profile=production,
        trunk_production_state="trunk_state",
        branch_production_states={"branch_0": "branch_0_state", "branch_1": "branch_1_state"},
        seed=41,
        poisson_observed=False,
    )

    segments = result["uns"]["segment_time_courses"]
    assert np.allclose(segments["trunk"]["alpha"][:, 0], 0.4)
    assert np.allclose(segments["trunk"]["alpha"][:, 1], 0.5)
    assert np.allclose(segments["branch_0"]["alpha"][:, 0], 1.2)
    assert np.allclose(segments["branch_0"]["alpha"][:, 1], 0.2)
    assert np.allclose(segments["branch_1"]["alpha"][:, 0], 0.1)
    assert np.allclose(segments["branch_1"]["alpha"][:, 1], 1.3)
    assert result["uns"]["simulation_config"]["production_profile"] is True


def test_bifurcation_interpolation_is_optional_not_default():
    production = StateProductionProfile(
        pd.DataFrame(
            {
                "g0": [0.4, 1.2, 0.1],
                "g1": [0.5, 0.2, 1.3],
            },
            index=["trunk_state", "branch_0_state", "branch_1_state"],
        )
    )

    result = simulate_bifurcation(
        _small_grn(),
        n_trunk_cells=6,
        n_branch_cells={"branch_0": 6, "branch_1": 6},
        trunk_time=0.8,
        branch_time=0.8,
        dt=0.04,
        production_profile=production,
        trunk_production_state="trunk_state",
        branch_production_states={"branch_0": "branch_0_state", "branch_1": "branch_1_state"},
        interpolate_production=True,
        seed=43,
        poisson_observed=False,
    )

    segments = result["uns"]["segment_time_courses"]
    assert np.isclose(segments["branch_0"]["alpha"][0, 0], 0.4)
    assert np.isclose(segments["branch_0"]["alpha"][-1, 0], 1.2)
    assert np.isclose(segments["branch_1"]["alpha"][0, 1], 0.5)
    assert np.isclose(segments["branch_1"]["alpha"][-1, 1], 1.3)
    assert not np.allclose(segments["branch_0"]["alpha"][0], segments["branch_0"]["alpha"][-1])


def test_bifurcation_requires_known_production_states():
    production = StateProductionProfile(pd.DataFrame({"g0": [0.4], "g1": [0.5]}, index=["trunk_state"]))

    with pytest.raises(ValueError, match="unknown production state"):
        simulate_bifurcation(
            _small_grn(),
            production_profile=production,
            trunk_production_state="trunk_state",
            branch_production_states={"branch_0": "missing", "branch_1": "missing"},
        )


def test_bifurcation_can_auto_calibrate_missing_half_response():
    production = StateProductionProfile(
        pd.DataFrame(
            {
                "g0": [0.4, 1.2, 0.1],
                "g1": [0.5, 0.2, 1.3],
            },
            index=["trunk_state", "branch_0_state", "branch_1_state"],
        )
    )

    result = simulate_bifurcation(
        _small_grn_missing_half_response(),
        n_trunk_cells=6,
        n_branch_cells={"branch_0": 6, "branch_1": 6},
        trunk_time=0.8,
        branch_time=0.8,
        dt=0.04,
        production_profile=production,
        trunk_production_state="trunk_state",
        branch_production_states={"branch_0": "branch_0_state", "branch_1": "branch_1_state"},
        auto_calibrate_half_response="if_missing",
        seed=59,
        poisson_observed=False,
    )

    assert result["uns"]["grn_calibration"]["calibration_method"] == "levelwise_state_mean"
    assert result["uns"]["grn_calibration"]["thresholds_filled_count"] == 2
    assert result["uns"]["simulation_config"]["auto_calibrate_half_response"] == "if_missing"
    assert pd.DataFrame(result["uns"]["true_grn"])["half_response"].notna().all()


def test_bifurcation_regulator_activity_modes_propagate_to_trunk_and_branches():
    genes = ["g0", "g1"]
    grn = GRN.from_dataframe(
        pd.DataFrame(
            {
                "regulator": ["g0"],
                "target": ["g1"],
                "K": [1.0],
                "sign": ["activation"],
                "half_response": [1.0],
                "hill_coefficient": [1.0],
            }
        ),
        genes=genes,
    )
    kwargs = dict(
        n_trunk_cells=2,
        n_branch_cells={"branch_0": 2, "branch_1": 2},
        trunk_time=0.1,
        branch_time=0.1,
        dt=0.1,
        beta=np.array([0.1, 0.1]),
        gamma=np.array([0.1, 0.1]),
        u0=np.array([2.0, 0.0]),
        s0=np.array([5.0, 0.0]),
        master_programs={"g0": 0.0},
        seed=67,
        poisson_observed=False,
    )

    spliced = simulate_bifurcation(grn, regulator_activity="spliced", **kwargs)
    unspliced = simulate_bifurcation(grn, regulator_activity="unspliced", **kwargs)
    total = simulate_bifurcation(grn, regulator_activity="total", **kwargs)

    expected_spliced = 5.0 / 6.0
    expected_unspliced = 2.0 / 3.0
    expected_total = 7.0 / 8.0

    for result, expected, mode in [
        (spliced, expected_spliced, "spliced"),
        (unspliced, expected_unspliced, "unspliced"),
        (total, expected_total, "total"),
    ]:
        segments = result["uns"]["segment_time_courses"]
        assert np.isclose(segments["trunk"]["alpha"][0, 1], expected)
        assert np.isclose(segments["branch_0"]["alpha"][0, 1], expected)
        assert np.isclose(segments["branch_1"]["alpha"][0, 1], expected)
        assert result["uns"]["simulation_config"]["regulator_activity"] == mode


def test_bifurcation_edge_contributions_have_expected_shape():
    result = simulate_bifurcation(
        _small_grn(),
        n_trunk_cells=5,
        n_branch_cells={"branch_0": 5, "branch_1": 5},
        trunk_time=0.8,
        branch_time=0.8,
        dt=0.04,
        seed=47,
        poisson_observed=False,
        return_edge_contributions=True,
    )

    assert result["edge_contributions"].shape == (15, result["uns"]["edge_metadata"].shape[0])


def test_bifurcation_output_contains_standardized_metadata():
    result = simulate_bifurcation(
        _small_grn(),
        n_trunk_cells=5,
        n_branch_cells={"branch_0": 5, "branch_1": 5},
        trunk_time=0.8,
        branch_time=0.8,
        dt=0.04,
        seed=53,
        poisson_observed=False,
        noise_model="poisson_capture",
    )

    assert set(["gene_role", "gene_class", "gene_level", "true_beta", "true_gamma"]).issubset(result["var"].columns)
    assert set(["true_grn", "grn_calibration", "kinetic_params", "simulation_config", "noise_config"]).issubset(
        result["uns"].keys()
    )
    assert result["uns"]["noise_config"]["noise_model"] == "poisson_capture"
