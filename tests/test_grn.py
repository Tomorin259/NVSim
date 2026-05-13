import numpy as np
import pandas as pd
import pytest

from nvsim.grn import (
    GRN,
    build_graph_levels,
    calibrate_grn_half_response,
    calibrate_half_response,
    estimate_state_mean_expression,
    identify_master_regulators,
    validate_grn,
)


def test_validate_grn_fills_defaults_and_normalizes_signs():
    edges = pd.DataFrame(
        {
            "regulator": ["g1", "g2"],
            "target": ["g3", "g3"],
            "K": [1.0, 0.5],
            "sign": ["+", "rep"],
        }
    )

    normalized = validate_grn(edges)

    assert list(normalized["sign"]) == ["activation", "repression"]
    assert list(normalized["K"]) == [1.0, 0.5]
    assert list(normalized["hill_coefficient"]) == [2.0, 2.0]
    assert normalized["half_response"].isna().all()
    assert set(normalized.columns) == {"regulator", "target", "sign", "K", "half_response", "hill_coefficient"}


def test_validate_grn_accepts_weight_alias_for_K():
    edges = pd.DataFrame(
        {
            "regulator": ["g1"],
            "target": ["g2"],
            "weight": [3.5],
            "sign": ["activation"],
        }
    )

    with pytest.warns(DeprecationWarning, match="legacy alias"):
        normalized = validate_grn(edges)

    assert normalized.loc[0, "K"] == 3.5
    assert "weight" not in normalized.columns


def test_validate_grn_threshold_alias_emits_deprecation_warning():
    edges = pd.DataFrame(
        {
            "regulator": ["g1"],
            "target": ["g2"],
            "K": [1.0],
            "sign": ["activation"],
            "threshold": [2.5],
            "hill_coefficient": [2.0],
        }
    )

    with pytest.warns(DeprecationWarning, match="legacy alias"):
        normalized = validate_grn(edges)

    assert normalized.loc[0, "half_response"] == 2.5
    assert "threshold" not in normalized.columns


def test_validate_grn_accepts_half_response_as_canonical_column():
    edges = pd.DataFrame(
        {
            "regulator": ["g1"],
            "target": ["g2"],
            "K": [1.0],
            "sign": ["activation"],
            "half_response": [2.5],
            "hill_coefficient": [2.0],
        }
    )

    normalized = validate_grn(edges)

    assert normalized.loc[0, "half_response"] == 2.5
    assert "threshold" not in normalized.columns


def test_validate_grn_keeps_missing_half_response_for_later_calibration():
    edges = pd.DataFrame(
        {
            "regulator": ["g1"],
            "target": ["g2"],
            "K": [1.0],
            "sign": ["activation"],
        }
    )
    normalized = validate_grn(edges)
    assert np.isnan(normalized.loc[0, "half_response"])


def test_validate_grn_rejects_negative_weights():
    edges = pd.DataFrame(
        {"regulator": ["g1"], "target": ["g2"], "K": [-1.0], "sign": ["activation"]}
    )

    with pytest.raises(ValueError, match="non-negative"):
        validate_grn(edges)


def test_grn_rejects_unknown_genes():
    edges = pd.DataFrame(
        {"regulator": ["g1"], "target": ["g2"], "K": [1.0], "sign": ["activation"]}
    )

    with pytest.raises(ValueError, match="absent"):
        GRN.from_dataframe(edges, genes=["g1"])


def test_calibrate_half_response_from_series_updates_canonical_columns():
    grn = GRN.from_dataframe(
        pd.DataFrame(
            {
                "regulator": ["g1", "g2"],
                "target": ["g3", "g3"],
                "K": [1.0, 0.5],
                "sign": ["activation", "repression"],
            }
        ),
        genes=["g1", "g2", "g3"],
    )

    calibrated = calibrate_half_response(grn, pd.Series({"g1": 2.5, "g2": 1.5}))
    edges = calibrated.to_dataframe()

    assert list(edges["half_response"]) == [2.5, 1.5]
    assert "threshold" not in edges.columns


def test_calibrate_half_response_from_dataframe_uses_column_means():
    grn = GRN.from_dataframe(
        pd.DataFrame(
            {
                "regulator": ["g1"],
                "target": ["g2"],
                "K": [1.0],
                "sign": ["activation"],
            }
        ),
        genes=["g1", "g2"],
    )

    expression = pd.DataFrame({"g1": [1.0, 3.0], "g2": [2.0, 2.0]})
    calibrated = calibrate_half_response(grn, expression)

    assert np.isclose(calibrated.to_dataframe().loc[0, "half_response"], 2.0)


def test_calibrate_half_response_rejects_missing_regulator_mean():
    grn = GRN.from_dataframe(
        pd.DataFrame(
            {
                "regulator": ["g1"],
                "target": ["g2"],
                "K": [1.0],
                "sign": ["activation"],
            }
        ),
        genes=["g1", "g2"],
    )

    with pytest.raises(ValueError, match="missing mean expression"):
        calibrate_half_response(grn, pd.Series({"g9": 1.0}))


def test_grn_preserves_explicit_master_regulators():
    grn = GRN.from_dataframe(
        pd.DataFrame(
            {
                "regulator": ["g0"],
                "target": ["g1"],
                "K": [1.0],
                "sign": ["activation"],
            }
        ),
        genes=["g0", "g1", "g2"],
        master_regulators=["g2"],
    )

    assert grn.master_regulators == ("g2",)


def test_identify_master_regulators_fallback_and_explicit_override():
    grn = GRN.from_dataframe(
        pd.DataFrame(
            {
                "regulator": ["g0", "g1"],
                "target": ["g2", "g3"],
                "K": [1.0, 1.0],
                "sign": ["activation", "activation"],
                "half_response": [1.0, 1.0],
                "hill_coefficient": [2.0, 2.0],
            }
        ),
        genes=["g0", "g1", "g2", "g3"],
    )
    masters, targets = identify_master_regulators(grn)
    assert masters == ("g0", "g1")
    assert targets == ("g2", "g3")

    masters2, targets2 = identify_master_regulators(grn, explicit_master_regulators=["g1"])
    assert masters2 == ("g1",)
    assert targets2 == ("g0", "g2", "g3")


def test_build_graph_levels_returns_levelwise_order_for_acyclic_grn():
    grn = GRN.from_dataframe(
        pd.DataFrame(
            {
                "regulator": ["g0", "g0", "g1"],
                "target": ["g1", "g2", "g3"],
                "K": [1.0, 1.0, 1.0],
                "sign": ["activation", "activation", "activation"],
                "half_response": [1.0, 1.0, 1.0],
                "hill_coefficient": [2.0, 2.0, 2.0],
            }
        ),
        genes=["g0", "g1", "g2", "g3"],
        master_regulators=["g0"],
    )

    levels = build_graph_levels(grn)
    assert levels["cyclic_or_acyclic"] == "acyclic"
    assert levels["gene_to_level"] == {"g0": 0, "g1": 1, "g2": 1, "g3": 2}


def test_build_graph_levels_marks_cycles_without_crashing():
    grn = GRN.from_dataframe(
        pd.DataFrame(
            {
                "regulator": ["g0", "g1"],
                "target": ["g1", "g0"],
                "K": [1.0, 1.0],
                "sign": ["activation", "activation"],
                "half_response": [1.0, 1.0],
                "hill_coefficient": [2.0, 2.0],
            }
        ),
        genes=["g0", "g1"],
    )

    levels = build_graph_levels(grn)
    assert levels["cyclic_or_acyclic"] == "cyclic"
    assert set(levels["unresolved_genes"]) == {"g0", "g1"}
    assert levels["warnings"]


def test_estimate_state_mean_expression_propagates_levelwise_for_acyclic_grn():
    grn = GRN.from_dataframe(
        pd.DataFrame(
            {
                "regulator": ["g0"],
                "target": ["g1"],
                "K": [2.0],
                "sign": ["activation"],
                "half_response": [1.0],
                "hill_coefficient": [1.0],
            }
        ),
        genes=["g0", "g1"],
        master_regulators=["g0"],
    )
    production = pd.DataFrame({"g0": [1.0, 3.0]}, index=["bin_0", "bin_1"])
    means, meta = estimate_state_mean_expression(grn, production)
    assert meta["cyclic_or_acyclic"] == "acyclic"
    assert np.allclose(means.loc[:, "g0"], [1.0, 3.0])
    assert np.allclose(means.loc[:, "g1"], [1.0, 1.5])


def test_calibrate_grn_half_response_fills_missing_half_response():
    grn = GRN.from_dataframe(
        pd.DataFrame(
            {
                "regulator": ["g0"],
                "target": ["g1"],
                "K": [2.0],
                "sign": ["activation"],
                "hill_coefficient": [1.0],
            }
        ),
        genes=["g0", "g1"],
        master_regulators=["g0"],
    )
    production = pd.DataFrame({"g0": [1.0, 3.0]}, index=["bin_0", "bin_1"])
    calibrated, meta = calibrate_grn_half_response(grn, production)
    edges = calibrated.to_dataframe()
    assert np.isclose(edges.loc[0, "half_response"], 2.0)
    assert "threshold" not in edges.columns
    assert meta["half_responses_filled_count"] == 1


def test_calibrate_grn_half_response_uses_fallback_for_cyclic_grn():
    grn = GRN.from_dataframe(
        pd.DataFrame(
            {
                "regulator": ["g0", "g1"],
                "target": ["g1", "g0"],
                "K": [1.0, 1.0],
                "sign": ["activation", "activation"],
                "hill_coefficient": [1.0, 1.0],
            }
        ),
        genes=["g0", "g1"],
        master_regulators=["g0"],
    )
    production = pd.DataFrame({"g0": [2.0]}, index=["bin_0"])
    calibrated, meta = calibrate_grn_half_response(grn, production, fallback_half_response=1.5)
    edges = calibrated.to_dataframe()
    assert meta["actual_calibration"] == "cyclic"
    assert meta["converged"] is True
    assert np.all(np.isfinite(edges["half_response"]))
    assert np.all(edges["half_response"] > 0)
