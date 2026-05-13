from __future__ import annotations

from argparse import Namespace

import pandas as pd

from nvsim.production import StateProductionProfile
from scripts import sergio_1200g_ds3_strong_bifurcation as runner


def _profile() -> StateProductionProfile:
    return StateProductionProfile(
        pd.DataFrame(
            {
                "m0": [0.0, 1.0, 4.0, 3.0],
                "m1": [0.0, 1.0, 0.0, 3.0],
            },
            index=["bin_0", "bin_1", "bin_2", "bin_3"],
        )
    )


def test_resolve_simulator_accepts_aliases():
    assert runner._resolve_simulator("linear") == "linear"
    assert runner._resolve_simulator("branch") == "bifurcation"
    assert runner._resolve_simulator("bifurcation") == "bifurcation"


def test_select_farthest_child_state_from_parent():
    profile = _profile()
    states = list(profile.states)
    child, meta = runner._select_farthest_child_state(profile, parent_state="bin_0", allowed_states=states)

    assert child == "bin_3"
    assert meta["selected_state"] == "bin_3"
    assert meta["selection_method"] == "farthest_from_parent_master_production_euclidean_distance"


def test_resolve_linear_child_state_prefers_auto_when_missing():
    profile = _profile()
    states = list(profile.states)
    args = Namespace(
        trunk_state="bin_0",
        child_state=None,
        no_auto_select_child_state=False,
    )

    child, meta = runner._resolve_linear_child_state(args, states, profile)

    assert child == "bin_3"
    assert meta["selected_state"] == "bin_3"


def test_resolve_linear_child_state_uses_explicit_when_provided():
    profile = _profile()
    states = list(profile.states)
    args = Namespace(
        trunk_state="bin_0",
        child_state="bin_2",
        no_auto_select_child_state=False,
    )

    child, meta = runner._resolve_linear_child_state(args, states, profile)

    assert child == "bin_2"
    assert meta["selection_method"] == "explicit"


def test_resolve_branch_states_auto_selects_farthest_pair():
    profile = _profile()
    states = list(profile.states)
    args = Namespace(
        branch_0_state=None,
        branch_1_state=None,
        no_auto_select_branch_states=False,
    )

    mapping, meta = runner._resolve_branch_states(args, states, profile)

    assert mapping == {"branch_0": "bin_0", "branch_1": "bin_3"}
    assert meta["selected_pair"] == ["bin_0", "bin_3"]
