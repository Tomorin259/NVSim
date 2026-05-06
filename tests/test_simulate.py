import numpy as np
import pandas as pd

from nvsim.grn import GRN
from nvsim.production import StateProductionProfile
from nvsim.simulate import simulate_linear


def _small_grn():
    genes = ["g0", "g1", "g2", "g3"]
    edges = pd.DataFrame(
        {
            "regulator": ["g0", "g1"],
            "target": ["g2", "g3"],
            "weight": [0.7, 0.4],
            "sign": ["activation", "repression"],
            "threshold": [0.5, 0.5],
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

    assert set(["gene_role", "gene_class"]).issubset(var.columns)
    assert set(var["gene_role"]) == {"master_regulator", "target"}
    assert set(var["gene_class"]) == {"unassigned"}
    assert var.loc["g0", "gene_role"] == "master_regulator"
    assert var.loc["g1", "gene_role"] == "master_regulator"
    assert var.loc["g2", "gene_role"] == "target"
    assert var.loc["g3", "gene_role"] == "target"


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

    pytest = __import__("pytest")
    with pytest.raises(ValueError, match="production_state"):
        simulate_linear(grn, production_profile=production)


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
    assert set(["gene_role", "gene_class"]).issubset(adata.var.columns)
    assert set(["true_grn", "kinetic_params", "simulation_config"]).issubset(adata.uns.keys())


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
