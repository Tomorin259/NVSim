import numpy as np
import pandas as pd
import pytest

from nvsim.grn import GRN
from nvsim.production import StateProductionProfile
from nvsim.simulate import simulate_linear


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


def test_true_velocity_matches_formula_and_states_are_nonnegative():
    grn = _small_grn()
    result = simulate_linear(grn, n_cells=25, time_end=2.0, dt=0.02, seed=7, poisson_observed=False)
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
    result = simulate_linear(grn, n_cells=17, time_end=1.0, dt=0.05, seed=1)

    for layer in result["layers"].values():
        assert layer.shape == (17, 4)
    assert result["obs"].shape[0] == 17
    assert result["var"].shape[0] == 4
    assert set(["pseudotime", "branch"]).issubset(result["obs"].columns)


def test_var_metadata_distinguishes_gene_role_from_gene_class():
    grn = _small_grn()
    result = simulate_linear(grn, n_cells=12, time_end=1.0, dt=0.05, seed=13, poisson_observed=False)
    var = result["var"]

    assert set(["gene_role", "gene_class", "gene_level", "true_beta", "true_gamma"]).issubset(var.columns)
    assert set(var["gene_role"]) == {"master_regulator", "target"}
    assert set(var["gene_class"]) == {"unassigned"}
    assert var.loc["g0", "gene_role"] == "master_regulator"
    assert var.loc["g1", "gene_role"] == "master_regulator"
    assert var.loc["g2", "gene_role"] == "target"
    assert var.loc["g3", "gene_role"] == "target"
    assert int(var.loc["g0", "gene_level"]) == 0
    assert int(var.loc["g2", "gene_level"]) == 1


def test_explicit_master_regulators_override_topology_inference():
    grn = GRN.from_dataframe(
        pd.DataFrame(
            {
                "regulator": ["g0", "g1"],
                "target": ["g1", "g2"],
                "K": [0.5, 0.5],
                "sign": ["activation", "activation"],
                "half_response": [1.0, 1.0],
                "hill_coefficient": [1.0, 1.0],
            }
        ),
        genes=["g0", "g1", "g2"],
    )
    production = StateProductionProfile(pd.DataFrame({"g1": [1.5]}, index=["bin_0"]))
    result = simulate_linear(
        grn,
        n_cells=8,
        time_end=0.8,
        dt=0.05,
        master_regulators=["g1"],
        production_profile=production,
        production_state="bin_0",
        seed=19,
        poisson_observed=False,
    )

    assert np.allclose(result["layers"]["true_alpha"][:, 1], 1.5)
    assert result["var"].loc["g1", "gene_role"] == "master_regulator"
    assert result["var"].loc["g0", "gene_role"] == "target"


def test_no_incoming_edge_inference_still_works_without_explicit_masters():
    grn = _small_grn()
    result = simulate_linear(grn, n_cells=8, time_end=0.8, dt=0.05, seed=23, poisson_observed=False)

    assert result["var"].loc["g0", "gene_role"] == "master_regulator"
    assert result["var"].loc["g1", "gene_role"] == "master_regulator"


def test_same_seed_reproduces_sampled_cells():
    grn = _small_grn()
    result1 = simulate_linear(grn, n_cells=20, time_end=1.5, dt=0.03, seed=11)
    result2 = simulate_linear(grn, n_cells=20, time_end=1.5, dt=0.03, seed=11)

    assert np.array_equal(result1["obs"]["time_index"].to_numpy(), result2["obs"]["time_index"].to_numpy())
    assert np.allclose(result1["layers"]["true_spliced"], result2["layers"]["true_spliced"])
    assert np.array_equal(result1["layers"]["spliced"], result2["layers"]["spliced"])


def test_observed_layers_are_separate_from_true_layers():
    grn = _small_grn()
    result = simulate_linear(grn, n_cells=30, time_end=2.0, dt=0.02, seed=3, capture_rate=0.3)
    layers = result["layers"]

    assert layers["spliced"] is not layers["true_spliced"]
    assert layers["unspliced"] is not layers["true_unspliced"]
    assert layers["spliced"].shape == layers["true_spliced"].shape
    assert not np.shares_memory(layers["spliced"], layers["true_spliced"])


def test_linear_simulation_can_use_state_production_profile():
    grn = _small_grn()
    production = StateProductionProfile(pd.DataFrame({"g0": [1.25], "g1": [0.75]}, index=["bin_0"]))
    result = simulate_linear(
        grn,
        n_cells=10,
        time_end=1.0,
        dt=0.05,
        production_profile=production,
        production_state="bin_0",
        master_programs={"g0": 99.0, "g1": 99.0},
        seed=17,
        poisson_observed=False,
    )

    assert result["uns"]["simulation_config"]["production_profile"] is True
    assert result["uns"]["simulation_config"]["production_state"] == "bin_0"
    assert np.allclose(result["layers"]["true_alpha"][:, 0], 1.25)
    assert np.allclose(result["layers"]["true_alpha"][:, 1], 0.75)
    assert not np.allclose(result["layers"]["true_alpha"][:, 0], 99.0)


def test_linear_simulation_requires_state_for_production_profile():
    grn = _small_grn()
    production = StateProductionProfile(pd.DataFrame({"g0": [1.0], "g1": [1.0]}, index=["bin_0"]))

    with pytest.raises(ValueError, match="production_state"):
        simulate_linear(grn, production_profile=production)


def test_linear_simulation_still_requires_half_response_without_auto_calibration():
    grn = _small_grn_missing_half_response()
    production = StateProductionProfile(pd.DataFrame({"g0": [1.0], "g1": [1.0]}, index=["bin_0"]))

    with pytest.raises(ValueError, match="missing half_response"):
        simulate_linear(
            grn,
            n_cells=8,
            time_end=0.8,
            dt=0.05,
            production_profile=production,
            production_state="bin_0",
            seed=37,
            poisson_observed=False,
        )


def test_linear_simulation_can_auto_calibrate_missing_half_response():
    grn = _small_grn_missing_half_response()
    production = StateProductionProfile(
        pd.DataFrame(
            {"g0": [1.0, 1.5], "g1": [0.5, 1.25]},
            index=["bin_0", "bin_1"],
        )
    )
    result = simulate_linear(
        grn,
        n_cells=8,
        time_end=0.8,
        dt=0.05,
        production_profile=production,
        production_state="bin_0",
        auto_calibrate_half_response="if_missing",
        seed=41,
        poisson_observed=False,
    )

    assert result["uns"]["grn_calibration"]["calibration_method"] == "levelwise_state_mean"
    assert result["uns"]["grn_calibration"]["thresholds_filled_count"] == grn.edges.shape[0]
    assert result["uns"]["simulation_config"]["auto_calibrate_half_response"] == "if_missing"
    assert pd.DataFrame(result["uns"]["true_grn"])["half_response"].notna().all()


def test_target_leak_alpha_default_preserves_previous_behavior():
    grn = _small_grn()
    base = simulate_linear(grn, n_cells=10, time_end=0.8, dt=0.05, seed=29, poisson_observed=False)
    leak0 = simulate_linear(
        grn,
        n_cells=10,
        time_end=0.8,
        dt=0.05,
        seed=29,
        poisson_observed=False,
        target_leak_alpha=0.0,
    )

    assert np.allclose(base["layers"]["true_alpha"], leak0["layers"]["true_alpha"])


def test_linear_edge_contributions_have_expected_shape():
    grn = _small_grn()
    result = simulate_linear(
        grn,
        n_cells=9,
        time_end=0.8,
        dt=0.05,
        seed=31,
        poisson_observed=False,
        return_edge_contributions=True,
    )

    assert "edge_contributions" in result
    assert result["edge_contributions"].shape == (9, grn.edges.shape[0])
    assert "edge_metadata" in result["uns"]


def test_optional_anndata_export_contains_expected_fields():
    pytest = __import__("pytest")
    ad = pytest.importorskip("anndata")
    from nvsim.output import to_anndata

    grn = _small_grn()
    result = simulate_linear(grn, n_cells=8, time_end=1.0, dt=0.05, seed=5)
    adata = to_anndata(result)

    assert adata.n_obs == 8
    assert adata.n_vars == 4
    assert set([
        "unspliced",
        "spliced",
        "true_unspliced",
        "true_spliced",
        "true_velocity",
        "true_velocity_u",
        "true_alpha",
    ]).issubset(adata.layers.keys())
    assert set(["pseudotime", "branch"]).issubset(adata.obs.columns)
    assert set(["gene_role", "gene_class", "gene_level", "true_beta", "true_gamma"]).issubset(adata.var.columns)
    assert set(["true_grn", "kinetic_params", "simulation_config", "grn_calibration", "noise_config"]).issubset(adata.uns.keys())
    assert adata.layers["true_velocity_u"].shape == (8, 4)


def test_optional_anndata_export_can_write_h5ad(tmp_path):
    pytest = __import__("pytest")
    pytest.importorskip("anndata")
    from nvsim.output import to_anndata

    grn = _small_grn()
    result = simulate_linear(grn, n_cells=8, time_end=1.0, dt=0.05, seed=5)
    adata = to_anndata(result)
    path = tmp_path / "roundtrip.h5ad"
    adata.write_h5ad(path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_result_uns_contains_grn_and_noise_metadata():
    grn = _small_grn()
    result = simulate_linear(
        grn,
        n_cells=8,
        time_end=1.0,
        dt=0.05,
        seed=5,
        capture_rate=0.3,
        noise_model="binomial_capture",
    )

    assert set(["true_grn", "grn_calibration", "kinetic_params", "simulation_config", "noise_config"]).issubset(
        result["uns"].keys()
    )
    assert result["uns"]["noise_config"]["noise_model"] == "binomial_capture"
    assert result["var"]["true_beta"].shape[0] == 4
    assert result["var"]["true_gamma"].shape[0] == 4


def test_regulator_activity_modes_change_alpha_as_expected():
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
        n_cells=2,
        time_end=0.1,
        dt=0.1,
        u0=np.array([2.0, 0.0]),
        s0=np.array([5.0, 0.0]),
        master_programs={"g0": 0.0},
        seed=103,
        poisson_observed=False,
    )

    spliced = simulate_linear(grn, regulator_activity="spliced", **kwargs)
    unspliced = simulate_linear(grn, regulator_activity="unspliced", **kwargs)
    total = simulate_linear(grn, regulator_activity="total", **kwargs)

    assert np.isclose(spliced["layers"]["true_alpha"][0, 1], 5.0 / 6.0)
    assert np.isclose(unspliced["layers"]["true_alpha"][0, 1], 2.0 / 3.0)
    assert np.isclose(total["layers"]["true_alpha"][0, 1], 7.0 / 8.0)
    assert spliced["uns"]["simulation_config"]["regulator_activity"] == "spliced"
    assert unspliced["uns"]["simulation_config"]["regulator_activity"] == "unspliced"
    assert total["uns"]["simulation_config"]["regulator_activity"] == "total"


def test_invalid_regulator_activity_raises():
    grn = _small_grn()
    with pytest.raises(ValueError, match="regulator_activity"):
        simulate_linear(grn, n_cells=4, time_end=0.5, dt=0.1, regulator_activity="bad_mode")


def test_constant_alpha_linear_ode_approaches_steady_state():
    genes = ["g0"]
    grn = GRN.from_dataframe(
        pd.DataFrame(columns=["regulator", "target", "weight", "sign"]),
        genes=genes,
    )
    alpha = 2.0
    beta = np.array([1.0])
    gamma = np.array([0.5])
    time_end = 20.0
    dt = 0.02
    n_timepoints = int(np.ceil(time_end / dt)) + 1

    result = simulate_linear(
        grn,
        n_cells=n_timepoints,
        time_end=time_end,
        dt=dt,
        beta=beta,
        gamma=gamma,
        u0=np.array([0.0]),
        s0=np.array([0.0]),
        master_programs={"g0": alpha},
        seed=101,
        poisson_observed=False,
    )
    terminal = result["obs"]["pseudotime"].to_numpy().argmax()

    assert np.isclose(result["layers"]["true_unspliced"][terminal, 0], alpha / beta[0], atol=2e-2)
    assert np.isclose(result["layers"]["true_spliced"][terminal, 0], alpha / gamma[0], atol=2e-2)
