import numpy as np
import pandas as pd
import pytest

from nvsim.grn import GRN
from nvsim.production import StateProductionProfile
from nvsim.production import linear_decrease, linear_increase
from nvsim.simulate import simulate_bifurcation, simulate_bifurcation_legacy


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
    assert list(result["obs"].columns) == [
        "pseudotime",
        "local_time",
        "branch",
        "segment",
        "segment_time_index",
        "global_time_index",
        "time_index",
    ]
    assert result["obs"].shape[0] == n_cells
    assert list(result["obs"]["branch"].unique()) == ["trunk", "branch_0", "branch_1"]
    assert (result["obs"].loc[result["obs"]["branch"] == "trunk", "pseudotime"] <= 1.0).all()
    assert (result["obs"].loc[result["obs"]["branch"] != "trunk", "local_time"] >= 0.0).all()
    assert result["uns"]["simulation_config"]["time_index_scope"] == "segment_local"


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


def test_bifurcation_branch_initial_state_is_excluded_by_default():
    result = _result(seed=31)
    branch_obs = result["obs"].loc[result["obs"]["branch"] != "trunk"]
    assert (branch_obs["segment_time_index"] > 0).all()
    assert result["uns"]["simulation_config"]["include_branch_initial_state"] is False


def test_bifurcation_can_include_branch_initial_state_when_requested():
    result = simulate_bifurcation(
        _small_grn(),
        n_trunk_cells=4,
        n_branch_cells={"branch_0": 4, "branch_1": 4},
        trunk_time=0.4,
        branch_time=0.4,
        dt=0.2,
        include_branch_initial_state=True,
        seed=17,
        poisson_observed=False,
        allow_snapshot_replacement=True,
    )
    branch_obs = result["obs"].loc[result["obs"]["branch"] != "trunk"]
    assert (branch_obs["segment_time_index"] == 0).any()
    assert result["uns"]["simulation_config"]["include_branch_initial_state"] is True


def test_bifurcation_global_time_index_maps_to_time_grid():
    result = _result(seed=31)
    obs = result["obs"]
    grid = result["time_grid"].copy().set_index("global_time_index")
    sample = obs.iloc[0]
    mapped = grid.loc[int(sample["global_time_index"])]
    assert np.isclose(mapped["pseudotime"], sample["pseudotime"])
    assert np.isclose(mapped["local_time"], sample["local_time"])
    assert mapped["branch"] == sample["branch"]


def test_bifurcation_segment_time_index_can_repeat_but_global_time_index_need_not_match():
    result = _result(seed=31)
    obs = result["obs"]
    branch0 = obs.loc[obs["branch"] == "branch_0", "segment_time_index"]
    branch1 = obs.loc[obs["branch"] == "branch_1", "segment_time_index"]
    overlap = set(branch0.tolist()).intersection(set(branch1.tolist()))
    assert overlap
    branch0_global = set(obs.loc[obs["branch"] == "branch_0", "global_time_index"].tolist())
    branch1_global = set(obs.loc[obs["branch"] == "branch_1", "global_time_index"].tolist())
    assert branch0_global.isdisjoint(branch1_global)


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
        trunk_state="trunk_state",
        branch_child_states={"branch_0": "branch_0_state", "branch_1": "branch_1_state"},
        transition_schedule="step",
        transition_midpoint=0.0,
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
        trunk_state="trunk_state",
        branch_child_states={"branch_0": "branch_0_state", "branch_1": "branch_1_state"},
        transition_schedule="linear",
        seed=43,
        poisson_observed=False,
    )

    segments = result["uns"]["segment_time_courses"]
    assert np.isclose(segments["branch_0"]["alpha"][0, 0], 0.4)
    assert np.isclose(segments["branch_0"]["alpha"][-1, 0], 1.2)
    assert np.isclose(segments["branch_1"]["alpha"][0, 1], 0.5)
    assert np.isclose(segments["branch_1"]["alpha"][-1, 1], 1.3)
    assert not np.allclose(segments["branch_0"]["alpha"][0], segments["branch_0"]["alpha"][-1])


def test_bifurcation_state_anchor_transitions_to_child_states_and_inherits_trunk():
    production = StateProductionProfile(
        pd.DataFrame(
            {
                "g0": [0.4, 1.4, 0.1],
                "g1": [0.5, 0.2, 1.5],
            },
            index=["progenitor", "lineage_A", "lineage_B"],
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
        trunk_state="progenitor",
        branch_child_states=("lineage_A", "lineage_B"),
        alpha_source_mode="state_anchor",
        transition_schedule="sigmoid",
        seed=43,
        poisson_observed=False,
    )

    segments = result["uns"]["segment_time_courses"]
    inheritance = result["uns"]["branch_inheritance"]
    assert np.allclose(inheritance["branch_0_initial_u"], inheritance["trunk_terminal_u"])
    assert np.allclose(inheritance["branch_1_initial_s"], inheritance["trunk_terminal_s"])
    assert np.isclose(segments["branch_0"]["alpha"][0, 0], 0.4)
    assert np.isclose(segments["branch_0"]["alpha"][-1, 0], 1.4)
    assert np.isclose(segments["branch_1"]["alpha"][0, 1], 0.5)
    assert np.isclose(segments["branch_1"]["alpha"][-1, 1], 1.5)
    config = result["uns"]["simulation_config"]
    assert config["alpha_source_mode"] == "state_anchor"
    assert config["trunk_state"] == "progenitor"
    assert config["branch_child_states"] == {"branch_0": "lineage_A", "branch_1": "lineage_B"}
    assert config["transition_schedule"] == "sigmoid"
    assert config["branch_divergence_configured"] is True
    assert config["branch_divergence_source"] == "state_anchor"


def test_bifurcation_state_anchor_requires_existing_states():
    production = StateProductionProfile(pd.DataFrame({"g0": [0.4], "g1": [0.5]}, index=["progenitor"]))

    with pytest.raises(ValueError, match="unknown production state"):
        simulate_bifurcation(
            _small_grn(),
            production_profile=production,
            trunk_state="progenitor",
            branch_child_states=("missing_A", "missing_B"),
            alpha_source_mode="state_anchor",
        )


def test_bifurcation_state_arguments_require_production_profile():
    with pytest.raises(ValueError, match="state_anchor state arguments require production_profile"):
        simulate_bifurcation(
            _small_grn(),
            trunk_state="progenitor",
            branch_child_states=("lineage_A", "lineage_B"),
        )


def test_bifurcation_legacy_wrapper_maps_to_canonical_state_anchor():
    production = StateProductionProfile(
        pd.DataFrame(
            {
                "g0": [0.4, 1.4, 0.1],
                "g1": [0.5, 0.2, 1.5],
            },
            index=["progenitor", "lineage_A", "lineage_B"],
        )
    )

    with pytest.warns(DeprecationWarning, match="Legacy bifurcation production-profile arguments"):
        result = simulate_bifurcation_legacy(
            _small_grn(),
            production_profile=production,
            trunk_production_state="progenitor",
            branch_production_states={"branch_0": "lineage_A", "branch_1": "lineage_B"},
            interpolate_production=True,
            seed=43,
            poisson_observed=False,
        )
    assert result["uns"]["simulation_config"]["trunk_state"] == "progenitor"
    assert result["uns"]["simulation_config"]["branch_child_states"] == {"branch_0": "lineage_A", "branch_1": "lineage_B"}
    assert result["uns"]["simulation_config"]["transition_schedule"] == "linear"


def test_bifurcation_subset_fill_uses_default_for_missing_master_alpha():
    production = StateProductionProfile(
        pd.DataFrame(
            {"g0": [0.4, 1.4, 0.1]},
            index=["progenitor", "lineage_A", "lineage_B"],
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
        trunk_state="progenitor",
        branch_child_states=("lineage_A", "lineage_B"),
        profile_gene_policy="subset_fill",
        default_master_alpha=0.9,
        seed=43,
        poisson_observed=False,
    )

    segments = result["uns"]["segment_time_courses"]
    assert np.allclose(segments["trunk"]["alpha"][:, 1], 0.9)
    assert np.allclose(segments["branch_0"]["alpha"][:, 1], 0.9)
    assert result["uns"]["simulation_config"]["profile_gene_policy"] == "subset_fill"


def test_bifurcation_requires_known_production_states():
    production = StateProductionProfile(pd.DataFrame({"g0": [0.4], "g1": [0.5]}, index=["trunk_state"]))

    with pytest.raises(ValueError, match="unknown production state"):
        simulate_bifurcation(
            _small_grn(),
            production_profile=production,
            trunk_state="trunk_state",
            branch_child_states={"branch_0": "missing", "branch_1": "missing"},
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
        trunk_state="trunk_state",
        branch_child_states={"branch_0": "branch_0_state", "branch_1": "branch_1_state"},
        transition_schedule="step",
        transition_midpoint=0.0,
        auto_calibrate_half_response="if_missing",
        seed=59,
        poisson_observed=False,
    )

    assert result["uns"]["grn_calibration"]["calibration_method"] == "levelwise_state_mean"
    assert result["uns"]["grn_calibration"]["half_responses_filled_count"] == 2
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
        n_branch_cells={"branch_0": 1, "branch_1": 1},
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

    def expected_alpha_from_segment(segment, mode):
        u = float(segment["u"][0, 0])
        s = float(segment["s"][0, 0])
        if mode == "spliced":
            activity = s
        elif mode == "unspliced":
            activity = u
        else:
            activity = u + s
        return activity / (1.0 + activity)

    for result, mode in [
        (spliced, "spliced"),
        (unspliced, "unspliced"),
        (total, "total"),
    ]:
        segments = result["uns"]["segment_time_courses"]
        assert np.isclose(segments["trunk"]["alpha"][0, 1], expected_alpha_from_segment(segments["trunk"], mode))
        assert np.isclose(
            segments["branch_0"]["alpha"][0, 1],
            expected_alpha_from_segment(segments["branch_0"], mode),
        )
        assert np.isclose(
            segments["branch_1"]["alpha"][0, 1],
            expected_alpha_from_segment(segments["branch_1"], mode),
        )
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
        capture_model="poisson_capture",
    )

    assert set(["gene_role", "gene_class", "gene_level", "true_beta", "true_gamma"]).issubset(result["var"].columns)
    assert set(["true_grn", "grn_calibration", "kinetic_params", "simulation_config", "noise_config"]).issubset(
        result["uns"].keys()
    )
    assert result["uns"]["noise_config"]["capture_model"] == "poisson_capture"


def test_bifurcation_capture_model_binomial_capture_records_canonical_noise_metadata():
    result = simulate_bifurcation(
        _small_grn(),
        n_trunk_cells=5,
        n_branch_cells={"branch_0": 5, "branch_1": 5},
        trunk_time=0.8,
        branch_time=0.8,
        dt=0.04,
        seed=53,
        capture_rate=0.4,
        capture_model="binomial_capture",
    )
    assert result["uns"]["simulation_config"]["capture_model"] == "binomial_capture"
    assert result["uns"]["noise_config"]["capture_model"] == "binomial_capture"


def test_bifurcation_without_branch_specific_difference_is_recorded_as_control():
    with pytest.warns(UserWarning, match="duplicated-branch control"):
        result = simulate_bifurcation(
            _small_grn(),
            n_trunk_cells=5,
            n_branch_cells={"branch_0": 5, "branch_1": 5},
            trunk_time=0.8,
            branch_time=0.8,
            dt=0.04,
            seed=53,
            poisson_observed=False,
        )
    config = result["uns"]["simulation_config"]
    assert config["branch_divergence_configured"] is False
    assert config["branch_divergence_source"] == "none"


def test_bifurcation_sampling_without_replacement_is_default():
    with pytest.raises(ValueError, match="n_cells exceeds available timepoints"):
        simulate_bifurcation(
            _small_grn(),
            n_trunk_cells=50,
            n_branch_cells={"branch_0": 50, "branch_1": 50},
            trunk_time=0.2,
            branch_time=0.2,
            dt=0.1,
            seed=11,
            poisson_observed=False,
        )


def test_bifurcation_sampling_replacement_can_be_enabled_and_is_recorded():
    result = simulate_bifurcation(
        _small_grn(),
        n_trunk_cells=50,
        n_branch_cells={"branch_0": 50, "branch_1": 50},
        trunk_time=0.2,
        branch_time=0.2,
        dt=0.1,
        seed=11,
        poisson_observed=False,
        allow_snapshot_replacement=True,
    )
    config = result["uns"]["simulation_config"]
    assert config["sampling_replace"] is True
    assert config["n_duplicate_snapshot_cells"] > 0
