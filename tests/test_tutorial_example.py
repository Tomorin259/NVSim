from pathlib import Path
import importlib.util


def _load_tutorial_module():
    root = Path(__file__).resolve().parents[1]
    tutorial_path = root / "examples" / "tutorial.py"
    spec = importlib.util.spec_from_file_location("nvsim_examples_tutorial", tutorial_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_tutorial_graph_examples_run_with_canonical_api():
    tutorial = _load_tutorial_module()

    linear = tutorial.run_linear_tutorial()
    branching = tutorial.run_branching_tutorial()

    assert linear["uns"]["simulation_config"]["capture_model"] == "poisson_capture"
    assert linear["uns"]["simulation_config"]["alpha_source_mode"] == "continuous_program"
    assert linear["uns"]["simulation_config"]["simulator"] == "graph"
    assert branching["uns"]["simulation_config"]["capture_model"] == "binomial_capture"
    assert branching["uns"]["simulation_config"]["alpha_source_mode"] == "state_anchor"
    assert branching["uns"]["simulation_config"]["root_states"] == ["root"]
    assert branching["uns"]["simulation_config"]["transition_schedule"] == "sigmoid"
    assert linear["layers"]["true_spliced"].shape[0] == sum(tutorial.linear_parameters()["n_cells_per_state"].values())
    assert set(branching["obs"]["state"].unique()) == {"root", "branch_A", "branch_B"}

