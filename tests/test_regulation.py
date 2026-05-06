import numpy as np
import pandas as pd

from nvsim.grn import GRN
from nvsim.regulation import compute_alpha, hill_activation, hill_repression


def test_hill_activation_and_repression_are_complements():
    x = np.array([0.0, 1.0, 10.0])
    act = hill_activation(x, half_response=1.0, hill_coefficient=2.0)
    rep = hill_repression(x, half_response=1.0, hill_coefficient=2.0)

    assert np.all(np.diff(act) >= 0)
    assert np.all(np.diff(rep) <= 0)
    assert np.allclose(act + rep, 1.0)
    assert act[0] == 0.0
    assert rep[0] == 1.0


def test_repression_uses_positive_weight_without_negative_sign():
    edges = pd.DataFrame(
        {
            "regulator": ["g1"],
            "target": ["g2"],
            "weight": [2.0],
            "sign": ["repression"],
            "half_response": [1.0],
            "hill_coefficient": [1.0],
        }
    )
    grn = GRN.from_dataframe(edges, genes=["g1", "g2"])

    alpha_low_regulator = compute_alpha({"g1": 0.0}, grn, basal_alpha=0.0)
    alpha_high_regulator = compute_alpha({"g1": 3.0}, grn, basal_alpha=0.0)

    assert alpha_low_regulator["g2"] == 2.0
    assert alpha_high_regulator["g2"] == 0.5
    assert alpha_high_regulator["g2"] >= 0.0
