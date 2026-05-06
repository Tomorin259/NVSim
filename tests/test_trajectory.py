from nvsim.trajectory import make_bifurcation_trajectory, make_linear_trajectory


def test_linear_trajectory_metadata():
    traj = make_linear_trajectory(4, branch="root")

    assert [
        "cell_id",
        "pseudotime",
        "branch",
        "parent_branch",
        "local_pseudotime",
        "local_time",
        "is_trunk",
    ] == list(traj.columns)
    assert traj["pseudotime"].iloc[0] == 0.0
    assert traj["pseudotime"].iloc[-1] == 1.0
    assert set(traj["branch"]) == {"root"}


def test_bifurcation_metadata_marks_independent_branches():
    traj = make_bifurcation_trajectory(
        n_trunk_cells=3,
        n_branch_cells={"left": 2, "right": 4},
        branches=("left", "right"),
    )

    assert len(traj) == 9
    assert bool(traj.loc[2, "branch_point"]) is True
    assert set(traj.loc[traj["branch"] == "left", "parent_branch"]) == {"trunk"}
    assert set(traj.loc[traj["branch"] == "right", "parent_branch"]) == {"trunk"}
    assert traj.loc[traj["branch"] == "left", "local_pseudotime"].iloc[0] == 0.0
    assert traj.loc[traj["branch"] == "right", "local_pseudotime"].iloc[0] == 0.0
    assert traj.loc[traj["branch"] == "left", "local_time"].iloc[0] == 0.0
    assert traj.loc[traj["branch"] == "right", "local_time"].iloc[0] == 0.0
    assert list(traj["sample_order"]) == list(range(len(traj)))
