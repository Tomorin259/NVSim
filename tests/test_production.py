import pandas as pd
import pytest

from nvsim.production import StateProductionProfile, transition_weight


def _profile():
    return StateProductionProfile(
        pd.DataFrame(
            {
                "g0": [0.5, 1.5],
                "g2": [2.0, 4.0],
            },
            index=["bin_0", "bin_1"],
        )
    )


def test_state_production_profile_returns_state_alpha():
    profile = _profile()

    alpha = profile.source_alpha("bin_0")

    assert profile.states == ("bin_0", "bin_1")
    assert profile.genes == ("g0", "g2")
    assert alpha.to_dict() == {"g0": 0.5, "g2": 2.0}


def test_state_production_profile_reindexes_to_full_gene_order():
    profile = _profile()

    alpha = profile.source_alpha("bin_1", genes=["g0", "g1", "g2"])

    assert alpha.to_dict() == {"g0": 1.5, "g1": 0.0, "g2": 4.0}


def test_state_production_profile_interpolated_warns_and_matches_linear_transition():
    profile = _profile()

    with pytest.warns(DeprecationWarning, match="deprecated"):
        alpha = profile.source_alpha_interpolated("bin_0", "bin_1", 0.25)

    assert alpha.to_dict() == {"g0": 0.75, "g2": 2.5}


def test_state_production_profile_transition_schedules():
    profile = _profile()

    assert transition_weight(0.25, schedule="step", midpoint=0.5) == 0.0
    assert transition_weight(0.75, schedule="step", midpoint=0.5) == 1.0
    assert transition_weight(0.25, schedule="linear") == 0.25

    start = profile.source_alpha_transition("bin_0", "bin_1", 0.0, schedule="sigmoid")
    middle = profile.source_alpha_transition("bin_0", "bin_1", 0.5, schedule="sigmoid")
    end = profile.source_alpha_transition("bin_0", "bin_1", 1.0, schedule="sigmoid")

    assert start.to_dict() == {"g0": 0.5, "g2": 2.0}
    assert end.to_dict() == {"g0": 1.5, "g2": 4.0}
    assert 0.5 < middle["g0"] < 1.5
    assert 2.0 < middle["g2"] < 4.0


def test_state_production_profile_validates_master_genes():
    profile = _profile()

    profile.validate_master_genes(["g0", "g2"])
    with pytest.raises(ValueError, match="missing"):
        profile.validate_master_genes(["g0", "g1", "g2"])
    with pytest.raises(ValueError, match="extra"):
        profile.validate_master_genes(["g0"])


def test_state_production_profile_rejects_bad_values():
    with pytest.raises(ValueError, match="non-negative"):
        StateProductionProfile(pd.DataFrame({"g0": [-0.1]}, index=["bin_0"]))
    with pytest.raises(ValueError, match="finite"):
        StateProductionProfile(pd.DataFrame({"g0": [float("inf")]}, index=["bin_0"]))
    with pytest.raises(ValueError, match="must not be empty"):
        StateProductionProfile(pd.DataFrame())


def test_state_production_profile_rejects_unknown_state_and_bad_fraction():
    profile = _profile()

    with pytest.raises(ValueError, match="unknown production state"):
        profile.source_alpha("bin_9")
    with pytest.raises(ValueError, match="unknown production state"):
        profile.validate_states(["bin_0", "bin_9"])
    with pytest.raises(ValueError, match="fraction"):
        profile.source_alpha_transition("bin_0", "bin_1", 1.1)
