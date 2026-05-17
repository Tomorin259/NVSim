from __future__ import annotations

import io

import pandas as pd

from nvsim.production import StateProductionProfile
from examples.prepare_trrust_mouse_dataset import (
    STANDARD_COLUMNS,
    build_signed_full_grn,
    build_small_benchmark_grn,
    choose_master_regulators,
    make_master_production_profile,
)


def _mock_rows() -> list[list[str]]:
    text = """TF\tTarget\tMode\tPMID
TfA\tGene1\tActivation\t111
TfA\tGene1\tActivation\t222
TfA\tGene2\tUnknown\t333
TfB\tGene1\tRepression\t444
TfC\tTfC\tActivation\t555
TfD\tGene3\tActivation\t666
TfD\tGene3\tRepression\t777
TfE\tGene3\tActivation\t888
TfF\tGene3\tActivation\t889
TfG\tGene3\tActivation\t890
TfH\tGene3\tActivation\t891
TfI\tGene3\tActivation\t892
TfA\tGene4\tActivation\t901
TfB\tGene4\tRepression\t902
"""
    return [line.split("\t") for line in io.StringIO(text).read().splitlines()]


def test_build_signed_full_grn_filters_unknown_conflicts_and_self_loops():
    full_df, stats = build_signed_full_grn(_mock_rows())

    assert list(full_df.columns) == STANDARD_COLUMNS
    assert set(full_df["sign"]) == {"+", "-"}
    assert stats["raw_edges"] == 14
    assert stats["signed_edges"] == 13
    assert stats["dropped_unknown_edges"] == 1
    assert stats["dropped_conflicting_pairs"] == 1
    assert stats["dropped_self_loops"] == 1
    assert ((full_df["regulator"] == "TfD") & (full_df["target"] == "Gene3")).sum() == 0
    tfa_gene1 = full_df.loc[(full_df["regulator"] == "TfA") & (full_df["target"] == "Gene1")].iloc[0]
    assert tfa_gene1["mode"] == "Activation"
    assert tfa_gene1["pmids"] == "111;222"
    assert float(tfa_gene1["K"]) == 1.0
    assert float(tfa_gene1["half_response"]) == 1.0
    assert float(tfa_gene1["hill_coefficient"]) == 2.0


def test_small_benchmark_respects_size_limits():
    full_df, _ = build_signed_full_grn(_mock_rows())
    small_df, meta = build_small_benchmark_grn(
        full_df,
        top_tfs=4,
        max_genes=6,
        max_edges=5,
        max_in_degree=3,
    )

    genes = set(small_df["regulator"]).union(set(small_df["target"]))
    indegree = small_df.groupby("target").size()
    assert len(genes) <= 6
    assert len(small_df) <= 5
    assert int(indegree.max()) <= 3
    assert meta["small_grn_max_in_degree"] <= 3


def test_small_benchmark_rejects_impossible_top_tf_gene_budget():
    full_df, _ = build_signed_full_grn(_mock_rows())
    try:
        build_small_benchmark_grn(full_df, top_tfs=8, max_genes=6, max_edges=5, max_in_degree=3)
    except ValueError as exc:
        assert "top_tfs cannot exceed max_genes" in str(exc)
    else:
        raise AssertionError("expected ValueError for top_tfs > max_genes")


def test_master_regulators_and_profile_are_readable_by_current_api():
    full_df, _ = build_signed_full_grn(_mock_rows())
    small_df, _ = build_small_benchmark_grn(full_df, top_tfs=8, max_genes=10, max_edges=20, max_in_degree=2)
    masters = choose_master_regulators(small_df, n_masters=4)
    profile = make_master_production_profile(masters, seed=7)

    assert 1 <= len(masters) <= 4
    assert list(profile.index) == ["root", "branch0", "branch1"]
    assert list(profile.columns) == masters
    state_profile = StateProductionProfile(profile)
    state_profile.validate_master_genes(masters)
    state_profile.validate_states(["root", "branch0", "branch1"])
    assert (profile.to_numpy() >= 0.3).all()
    assert (profile.to_numpy() <= 3.0).all()
