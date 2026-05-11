from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.celloracle_demo_regression import (
    _scale_edge_strengths,
    choose_master_regulators,
    export_expression_supported_grn,
    extract_target_to_tfs,
    fit_global_ridge_models,
    make_master_production_profile,
)


def _mock_base_grn() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "peak_id": ["p1", "p2", "p3", "p4"],
            "gene_short_name": ["GeneA", "GeneA", "GeneB", "TF1"],
            "TF1": [1, 0, 1, 1],
            "TF2": [0, 1, 1, 0],
            "GeneA": [0, 0, 1, 0],
            "NoiseTF": [1, 0, 0, 0],
        }
    )


def _mock_expression_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 80
    tf1 = rng.normal(size=n)
    tf2 = rng.normal(size=n)
    gene_a = 1.5 * tf1 - 0.7 * tf2 + rng.normal(scale=0.05, size=n)
    gene_b = -1.0 * tf1 + 0.8 * tf2 + rng.normal(scale=0.05, size=n)
    tf_self = rng.normal(size=n)
    return pd.DataFrame(
        {
            "TF1": tf1,
            "TF2": tf2,
            "GeneA": gene_a,
            "GeneB": gene_b,
            "TF_SELF": tf_self,
        },
        index=[f"cell_{i}" for i in range(n)],
    )


def _mock_args():
    class Args:
        min_tfs = 2
        min_target_var = 1e-4
        test_size = 0.2
        seed = 1
        n_jobs = 1
        quick = False
        quick_n_targets = 200
        min_r2 = 0.1
        min_pearson = 0.3
        max_targets = 500
        max_edges = 3000
        max_tfs_per_target = 5
        min_cells_per_cluster = 20

    return Args()


def test_extract_target_to_tfs_merges_peaks_filters_and_drops_self_loops():
    base_grn = _mock_base_grn()
    expressed = {"TF1", "TF2", "GeneA", "GeneB", "TF_SELF"}
    target_to_tfs, base_edges, stats = extract_target_to_tfs(base_grn, expressed)

    assert target_to_tfs["GeneA"] == ["TF1", "TF2"]
    assert target_to_tfs["GeneB"] == ["GeneA", "TF1", "TF2"]
    assert "NoiseTF" not in base_edges["source"].tolist()
    assert ("TF1", "TF1") not in set(zip(base_edges["source"], base_edges["target"]))
    assert stats["n_candidate_targets"] >= 2


def test_fit_global_ridge_models_runs_on_small_mock_matrix():
    expression_df = _mock_expression_df()
    target_to_tfs = {
        "GeneA": ["TF1", "TF2"],
        "GeneB": ["TF1", "TF2"],
    }
    cluster_labels = pd.Series(np.where(np.arange(expression_df.shape[0]) % 2 == 0, "A", "B"), index=expression_df.index)
    metrics_df, skipped_df = fit_global_ridge_models(expression_df, target_to_tfs, cluster_labels, _mock_args())

    assert skipped_df.empty
    assert set(metrics_df["target"]) == {"GeneA", "GeneB"}
    assert (metrics_df["r2"] > 0.8).all()
    assert (metrics_df["pearson"] > 0.9).all()


def test_exported_grn_has_nvsim_columns_and_signed_edges():
    metrics_df = pd.DataFrame(
        [
            {
                "target": "GeneA",
                "r2": 0.5,
                "pearson": 0.8,
                "spearman": 0.7,
                "mse": 0.1,
                "n_tfs": 2,
                "n_cells": 80,
                "best_alpha": 1.0,
                "target_mean": 1.0,
                "target_var": 0.2,
                "tfs": "TF1;TF2",
                "coefs": "1.5;-0.5",
                "cluster_labels_used": True,
            }
        ]
    )
    exported = export_expression_supported_grn(metrics_df, _mock_args())

    assert list(exported.columns) == [
        "regulator",
        "target",
        "sign",
        "K",
        "half_response",
        "hill_coefficient",
        "source",
        "evidence",
        "r2",
        "pearson",
        "spearman",
        "coef",
    ]
    assert set(exported["sign"]) == {"+", "-"}
    assert exported["K"].between(0.2, 2.0).all()


def test_scale_edge_strengths_maps_to_expected_range():
    values = np.array([0.1, 0.3, 0.9], dtype=float)
    scaled = _scale_edge_strengths(values)
    assert np.isclose(scaled.min(), 0.2)
    assert np.isclose(scaled.max(), 2.0)


def test_master_regulators_and_profile_generation():
    exported = pd.DataFrame(
        {
            "regulator": ["TF1", "TF1", "TF2"],
            "target": ["GeneA", "GeneB", "GeneA"],
            "sign": ["+", "-", "+"],
            "K": [1.0, 0.8, 0.6],
            "half_response": [1.0, 1.0, 1.0],
            "hill_coefficient": [2.0, 2.0, 2.0],
            "source": ["x", "x", "x"],
            "evidence": ["y", "y", "y"],
            "r2": [0.5, 0.4, 0.3],
            "pearson": [0.8, 0.7, 0.6],
            "spearman": [0.7, 0.6, 0.5],
            "coef": [1.2, -0.7, 0.4],
        }
    )
    expression_df = _mock_expression_df()[["TF1", "TF2"]]
    obs = pd.DataFrame(
        {
            "louvain_annot": ["MEP_0"] * 20 + ["Ery_0"] * 30 + ["Gran_0"] * 30,
            "dpt_pseudotime": np.linspace(0.0, 1.0, 80),
        },
        index=expression_df.index,
    )
    masters = choose_master_regulators(exported, n_masters=2)
    profile = make_master_production_profile(expression_df, obs, masters, "louvain_annot", seed=1)

    assert 1 <= len(masters) <= 2
    assert list(profile.index) == ["root", "branch0", "branch1"]
    assert list(profile.columns) == masters
    assert (profile.to_numpy() >= 0.3).all()
    assert (profile.to_numpy() <= 3.0).all()
