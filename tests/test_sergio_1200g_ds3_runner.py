from __future__ import annotations

import pandas as pd

from nvsim.modes import branching_graph, path_graph
from nvsim.production import StateProductionProfile
from scripts import sergio_1200g_ds3_strong_branching as runner


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


def test_select_farthest_child_state_from_parent():
    profile = _profile()
    states = list(profile.states)
    child, meta = runner._select_farthest_child_state(profile, parent_state="bin_0", allowed_states=states)

    assert child == "bin_3"
    assert meta["selected_state"] == "bin_3"
    assert meta["selection_method"] == "farthest_from_parent_master_production_euclidean_distance"


def test_select_farthest_pair_for_branching_graph():
    profile = _profile()
    mapping, meta = runner._select_branch_leaf_states(profile)

    assert mapping == {"branch_0": "bin_0", "branch_1": "bin_3"}
    assert meta["selected_pair"] == ["bin_0", "bin_3"]


def test_build_path_graph_uses_root_and_terminal_states():
    graph = runner.build_path_template("bin_0", "bin_3")
    assert graph.topological_order() == ("bin_0", "bin_3")


def test_build_branching_graph_uses_root_and_leaf_states():
    graph = runner.build_branching_template("bin_0", {"branch_0": "bin_2", "branch_1": "bin_3"})
    assert set(graph.children_of("bin_0")) == {"bin_2", "bin_3"}
    assert graph.parent_of("bin_2") == "bin_0"
