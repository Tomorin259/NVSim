from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from nvsim.grn import GRN
from nvsim.output import to_anndata
from nvsim.plotting import (
    compute_pca_embedding,
    plot_embedding_by_pseudotime,
    plot_embedding_with_velocity,
    plot_gene_dynamics_over_pseudotime,
    plot_phase_portrait_gallery,
    plot_phase_portrait,
    select_representative_genes_by_dynamics,
)
from nvsim.programs import linear_decrease, linear_increase
from nvsim.simulate import simulate_bifurcation, simulate_linear


def _result():
    grn = GRN.from_dataframe(
        pd.DataFrame(
            {
                "regulator": ["g0"],
                "target": ["g1"],
                "weight": [0.8],
                "sign": ["activation"],
                "half_response": [0.5],
            }
        ),
        genes=["g0", "g1", "g2"],
    )
    return simulate_linear(grn, n_cells=12, time_end=1.0, dt=0.05, seed=21)


def test_plotting_functions_save_files(tmp_path: Path):
    result = _result()
    embedding, components = compute_pca_embedding(result, layer_preference="observed")
    assert embedding.shape == (12, 2)
    assert components.shape == (2, 3)

    paths = [
        tmp_path / "embedding_by_pseudotime.png",
        tmp_path / "phase_portrait_gene0_true.png",
        tmp_path / "phase_portrait_gene0_observed.png",
        tmp_path / "gene_dynamics_gene0.png",
    ]
    plot_embedding_by_pseudotime(result, embedding=embedding, output_path=paths[0])
    plot_phase_portrait(result, "g0", mode="true", output_path=paths[1])
    plot_phase_portrait(result, "g0", mode="observed", output_path=paths[2])
    plot_gene_dynamics_over_pseudotime(result, "g0", output_path=paths[3])
    plt.close("all")

    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_true_and_observed_pca_layer_preferences_run():
    result = _result()
    true_embedding, true_components = compute_pca_embedding(result, layer_preference="true")
    observed_embedding, observed_components = compute_pca_embedding(result, layer_preference="observed")

    assert true_embedding.shape == observed_embedding.shape == (12, 2)
    assert true_components.shape == observed_components.shape == (2, 3)


def test_velocity_plot_defaults_to_pca(tmp_path: Path):
    result = _result()
    path = tmp_path / "velocity.png"
    fig = plot_embedding_with_velocity(result, output_path=path)
    title = fig.axes[0].get_title()
    plt.close("all")

    assert title.startswith("PCA true spliced")
    assert path.exists()
    assert path.stat().st_size > 0


def test_plotting_handles_anndata_when_available(tmp_path: Path):
    pytest.importorskip("anndata")
    adata = to_anndata(_result())
    path = tmp_path / "adata_phase.png"
    plot_phase_portrait(adata, "g0", mode="true", output_path=path)
    plt.close("all")
    assert path.exists()
    assert path.stat().st_size > 0


def test_phase_portrait_gallery_saves_thumbnail_grid(tmp_path: Path):
    result = _result()
    path = tmp_path / "gallery.png"
    fig = plot_phase_portrait_gallery(result, mode="true", max_cols=2, output_path=path)
    plt.close(fig)

    assert path.exists()
    assert path.stat().st_size > 0


def test_dynamic_representative_selection_prefers_valid_edge_types():
    grn = GRN.from_dataframe(
        pd.DataFrame(
            {
                "regulator": ["g0", "g1"],
                "target": ["g2", "g3"],
                "weight": [0.8, 0.7],
                "sign": ["activation", "repression"],
                "half_response": [0.5, 0.5],
            }
        ),
        genes=["g0", "g1", "g2", "g3"],
    )
    result = simulate_bifurcation(
        grn,
        n_trunk_cells=8,
        n_branch_cells={"branch_0": 8, "branch_1": 8},
        trunk_time=1.0,
        branch_time=1.0,
        dt=0.05,
        master_programs={"g0": linear_increase(0.2, 0.7), "g1": linear_increase(0.3, 0.6)},
        branch_master_programs={
            "branch_0": {"g0": linear_increase(0.7, 1.4), "g1": linear_decrease(0.6, 0.1)},
            "branch_1": {"g0": linear_decrease(0.7, 0.1), "g1": linear_increase(0.6, 1.2)},
        },
        seed=9,
        poisson_observed=False,
    )

    selection = select_representative_genes_by_dynamics(result, grn)
    selected = selection["genes"]

    assert set(selected).issuperset({"master", "activation_target", "repression_target"})
    assert selected["master"] in grn.genes
    assert selected["activation_target"] == "g2"
    assert selected["repression_target"] == "g3"
    assert selection["alpha_differences"]["master"] >= 0.0
    assert set(selection["edges"]["activation_target"]["sign"]) == {"activation"}
    assert set(selection["edges"]["repression_target"]["sign"]) == {"repression"}
