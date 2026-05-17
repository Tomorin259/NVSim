from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from nvsim.grn import GRN
from nvsim.modes import branching_graph, path_graph
from nvsim.noise import apply_observation
from nvsim.output import to_anndata
from nvsim.plotting import (
    embed,
    plot_gene_dynamics,
    plot_phase_gallery,
    plot_phase_portrait,
    prepare_adata,
    select_genes,
)
from nvsim.velocity_plotting import prepare_velocity_adata
from nvsim.production import StateProductionProfile, linear_decrease, linear_increase
from nvsim.simulate import simulate


def _linear_result():
    grn = GRN.from_dataframe(
        pd.DataFrame(
            {
                "regulator": ["g0"],
                "target": ["g1"],
                "K": [0.8],
                "sign": ["activation"],
                "half_response": [0.5],
            }
        ),
        genes=["g0", "g1", "g2"],
    )
    result = simulate(
        grn,
        graph=path_graph(["early", "late"]),
        alpha_source_mode="continuous_program",
        master_programs={"g0": linear_increase(0.2, 0.7)},
        n_cells_per_state={"early": 6, "late": 6},
        root_time=1.0,
        state_time={"early": 1.0, "late": 1.0},
        dt=0.05,
        seed=21,
    )
    return apply_observation(result, seed=23, observation_sample=False)


def _branching_result():
    grn = GRN.from_dataframe(
        pd.DataFrame(
            {
                "regulator": ["g0", "g1"],
                "target": ["g2", "g3"],
                "K": [0.8, 0.7],
                "sign": ["activation", "repression"],
                "half_response": [0.5, 0.5],
            }
        ),
        genes=["g0", "g1", "g2", "g3"],
    )
    profile = StateProductionProfile(
        pd.DataFrame(
            {
                "g0": [0.5, 1.3, 0.2],
                "g1": [0.4, 0.1, 1.1],
            },
            index=["root", "branch_0", "branch_1"],
        )
    )
    result = simulate(
        grn,
        graph=branching_graph("root", ["branch_0", "branch_1"]),
        production_profile=profile,
        alpha_source_mode="state_anchor",
        n_cells_per_state={"root": 8, "branch_0": 8, "branch_1": 8},
        root_time=1.0,
        state_time={"root": 1.0, "branch_0": 1.0, "branch_1": 1.0},
        dt=0.05,
        seed=9,
    )
    return apply_observation(result, seed=11, observation_sample=False)


def test_prepare_adata_true_layers_from_result_dict():
    result = _linear_result()
    adata = prepare_adata(result, expression_layer="true")

    assert np.allclose(adata.X, result["layers"]["true_spliced"])
    assert np.allclose(adata.layers["spliced"], result["layers"]["true_spliced"])
    assert np.allclose(adata.layers["unspliced"], result["layers"]["true_unspliced"])
    assert np.allclose(adata.layers["velocity"], result["layers"]["true_velocity"])
    assert adata.uns["nvsim_velocity_showcase"]["expression_layer"] == "true"


def test_prepare_adata_observed_layers_from_result_dict():
    result = _linear_result()
    adata = prepare_adata(result, expression_layer="observed")

    assert np.allclose(adata.X, result["layers"]["spliced"])
    assert np.allclose(adata.layers["spliced"], result["layers"]["spliced"])
    assert np.allclose(adata.layers["unspliced"], result["layers"]["unspliced"])
    assert np.allclose(adata.layers["velocity"], result["layers"]["true_velocity"])


def test_prepare_adata_missing_layer_raises_clear_error():
    result = _linear_result()
    del result["layers"]["true_velocity"]

    with pytest.raises(KeyError, match="true_velocity"):
        prepare_adata(result, expression_layer="true")


def test_embed_smoke_when_scanpy_available():
    pytest.importorskip("scanpy")
    adata = prepare_adata(_linear_result(), expression_layer="true")
    embed(adata, n_pcs=20, n_neighbors=5, random_state=0)

    assert "X_pca" in adata.obsm
    assert "X_umap" in adata.obsm
    assert adata.obsm["X_umap"].shape == (adata.n_obs, 2)


def test_phase_portrait_uses_two_dimensional_gene_velocity(tmp_path: Path):
    result = _linear_result()
    adata = to_anndata(result)
    path = tmp_path / "phase.png"
    fig = plot_phase_portrait(adata, "g0", mode="true", output_path=path, show_velocity=True)
    plt.close(fig)

    assert path.exists()
    assert path.stat().st_size > 0
    assert "true_velocity" in adata.layers
    assert "true_velocity_u" in adata.layers


def test_observed_phase_portrait_uses_observed_layers():
    result = _linear_result()
    result["layers"]["spliced"] = result["layers"]["true_spliced"] + 100.0
    result["layers"]["unspliced"] = result["layers"]["true_unspliced"] + 200.0

    fig = plot_phase_portrait(result, "g0", mode="observed", show_velocity=False)
    offsets = np.asarray(fig.axes[0].collections[0].get_offsets(), dtype=float)
    plt.close(fig)

    assert np.allclose(offsets[:, 0], result["layers"]["spliced"][:, 0])
    assert np.allclose(offsets[:, 1], result["layers"]["unspliced"][:, 0])


def test_phase_portrait_accepts_explicit_layers_and_color_key():
    result = _linear_result()
    adata = to_anndata(result)
    adata.obs["clusters"] = pd.Categorical(["c0"] * adata.n_obs)
    adata.uns["clusters_colors"] = ["#1f77b4"]

    fig, ax = plt.subplots()
    ax = plot_phase_portrait(
        adata,
        "g0",
        ax=ax,
        layer_s="true_spliced",
        layer_u="true_unspliced",
        layer_v_s="true_velocity",
        layer_v_u="true_velocity_u",
        color_key="clusters",
    )
    plt.close(fig)

    assert ax.get_xlabel() == "Spliced (true_spliced)"
    assert ax.get_ylabel() == "Unspliced (true_unspliced)"


def test_phase_gallery_and_gene_dynamics_save_files(tmp_path: Path):
    result = _linear_result()
    paths = [
        tmp_path / "gallery.png",
        tmp_path / "gene_dynamics.png",
    ]
    fig = plot_phase_gallery(result, genes=["g0", "g1"], mode="true", max_cols=2, output_path=paths[0])
    plt.close(fig)
    fig = plot_gene_dynamics(result, genes=["g0", "g1"], output_path=paths[1])
    assert len(fig.axes) == 10
    plt.close(fig)

    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_select_genes_returns_valid_representatives():
    adata = prepare_adata(_branching_result(), expression_layer="true")
    selected = select_genes(adata, representative_genes="auto")

    assert 3 <= len(selected) <= 4
    assert set(selected).issubset(set(map(str, adata.var_names)))


def test_velocity_plotting_compatibility_shim_still_imports():
    result = _linear_result()
    adata = prepare_velocity_adata(result, expression_layer="true")

    assert np.allclose(adata.X, result["layers"]["true_spliced"])
