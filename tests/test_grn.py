import pandas as pd
import pytest

from nvsim.grn import GRN, validate_grn


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
    assert list(normalized["threshold"]) == [1.0, 1.0]


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
