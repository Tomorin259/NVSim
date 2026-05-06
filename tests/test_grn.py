import numpy as np
import pandas as pd
import pytest

from nvsim.grn import GRN, calibrate_half_response, validate_grn


def test_validate_grn_fills_defaults_and_normalizes_signs():
    edges = pd.DataFrame(
        {
            "regulator": ["g1", "g2"],
            "target": ["g3", "g3"],
            "weight": [1.0, 0.5],
            "sign": ["+", "rep"],
        }
    )

    normalized = validate_grn(edges)

    assert list(normalized["sign"]) == ["activation", "repression"]
    assert list(normalized["hill_coefficient"]) == [2.0, 2.0]
    assert list(normalized["half_response"]) == [1.0, 1.0]
    assert list(normalized["threshold"]) == [1.0, 1.0]


def test_validate_grn_accepts_half_response_and_keeps_threshold_alias():
    edges = pd.DataFrame(
        {
            "regulator": ["g1"],
            "target": ["g2"],
            "weight": [1.0],
            "sign": ["activation"],
            "half_response": [2.5],
            "hill_coefficient": [2.0],
        }
    )

    normalized = validate_grn(edges)

    assert normalized.loc[0, "half_response"] == 2.5
    assert normalized.loc[0, "threshold"] == 2.5


def test_validate_grn_rejects_negative_weights():
    edges = pd.DataFrame(
        {"regulator": ["g1"], "target": ["g2"], "weight": [-1.0], "sign": ["activation"]}
    )

    with pytest.raises(ValueError, match="non-negative"):
        validate_grn(edges)


def test_grn_rejects_unknown_genes():
    edges = pd.DataFrame(
        {"regulator": ["g1"], "target": ["g2"], "weight": [1.0], "sign": ["activation"]}
    )

    with pytest.raises(ValueError, match="absent"):
        GRN.from_dataframe(edges, genes=["g1"])


def test_calibrate_half_response_from_series_updates_alias_columns():
    grn = GRN.from_dataframe(
        pd.DataFrame(
            {
                "regulator": ["g1", "g2"],
                "target": ["g3", "g3"],
                "weight": [1.0, 0.5],
                "sign": ["activation", "repression"],
            }
        ),
        genes=["g1", "g2", "g3"],
    )

    calibrated = calibrate_half_response(grn, pd.Series({"g1": 2.5, "g2": 0.0}))
    edges = calibrated.to_dataframe()

    assert list(edges["half_response"]) == [2.5, 0.0]
    assert list(edges["threshold"]) == [2.5, 0.0]


def test_calibrate_half_response_from_dataframe_uses_column_means():
    grn = GRN.from_dataframe(
        pd.DataFrame(
            {
                "regulator": ["g1"],
                "target": ["g2"],
                "weight": [1.0],
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
                "weight": [1.0],
                "sign": ["activation"],
            }
        ),
        genes=["g1", "g2"],
    )

    with pytest.raises(ValueError, match="missing mean expression"):
        calibrate_half_response(grn, pd.Series({"g9": 1.0}))
