import numpy as np
import pandas as pd

from nvsim.grn import GRN
from nvsim.modes import path_graph
from nvsim.production import StateProductionProfile
from nvsim.simulate import simulate


def _small_grn():
    genes = ["g0", "g1", "g2", "g3"]
    edges = pd.DataFrame(
        {
            "regulator": ["g0", "g1"],
            "target": ["g2", "g3"],
            "K": [0.7, 0.4],
            "sign": ["activation", "repression"],
            "half_response": [0.5, 0.5],
            "hill_coefficient": [2.0, 2.0],
        }
    )
    return GRN.from_dataframe(edges, genes=genes)


def test_state_anchor_step_uses_child_alpha_throughout_child_state():
    grn = _small_grn()
    profile = StateProductionProfile(
        pd.DataFrame(
            {
                "g0": [0.2, 1.1],
                "g1": [0.4, 0.9],
            },
            index=["early", "late"],
        )
    )
    result = simulate(
        grn,
        graph=path_graph(["early", "late"]),
        alpha_source_mode="state_anchor",
        production_profile=profile,
        n_cells_per_state={"early": 6, "late": 6},
        root_time=1.0,
        state_time={"early": 1.0, "late": 1.0},
        dt=0.05,
        seed=19,
        poisson_observed=False,
        transition_schedule="step",
    )

    late_mask = (result["obs"]["state"] == "late").to_numpy()
    late_idx = np.flatnonzero(late_mask)
    g0_idx = list(result["var"].index).index("g0")
    g1_idx = list(result["var"].index).index("g1")
    alpha = result["layers"]["true_alpha"]

    assert np.allclose(alpha[late_idx, g0_idx], 1.1)
    assert np.allclose(alpha[late_idx, g1_idx], 0.9)
