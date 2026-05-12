import numpy as np
import pandas as pd
import pytest

from nvsim.grn import GRN
from nvsim.simulate import simulate_linear
from nvsim.velocity_plotting import prepare_velocity_adata, run_scanpy_embedding


def _result():
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
    return simulate_linear(grn, n_cells=12, time_end=1.0, dt=0.05, seed=21, poisson_observed=False)


def test_prepare_velocity_adata_true_layers_from_result_dict():
    result = _result()
    adata = prepare_velocity_adata(result, expression_layer="true")

    assert np.allclose(adata.X, result["layers"]["true_spliced"])
    assert np.allclose(adata.layers["spliced"], result["layers"]["true_spliced"])
    assert np.allclose(adata.layers["unspliced"], result["layers"]["true_unspliced"])
    assert np.allclose(adata.layers["velocity"], result["layers"]["true_velocity"])
    assert adata.uns["nvsim_velocity_showcase"]["expression_layer"] == "true"


def test_prepare_velocity_adata_observed_layers_from_result_dict():
    result = _result()
    adata = prepare_velocity_adata(result, expression_layer="observed")

    assert np.allclose(adata.X, result["layers"]["spliced"])
    assert np.allclose(adata.layers["spliced"], result["layers"]["spliced"])
    assert np.allclose(adata.layers["unspliced"], result["layers"]["unspliced"])
    assert np.allclose(adata.layers["velocity"], result["layers"]["true_velocity"])


def test_prepare_velocity_adata_missing_layer_raises_clear_error():
    result = _result()
    del result["layers"]["true_velocity"]

    with pytest.raises(KeyError, match="true_velocity"):
        prepare_velocity_adata(result, expression_layer="true")


def test_run_scanpy_embedding_smoke_when_available():
    pytest.importorskip("scanpy")
    adata = prepare_velocity_adata(_result(), expression_layer="true")
    run_scanpy_embedding(adata, n_pcs=20, n_neighbors=5, random_state=0)

    assert "X_pca" in adata.obsm
    assert "X_umap" in adata.obsm
    assert adata.obsm["X_umap"].shape == (adata.n_obs, 2)
