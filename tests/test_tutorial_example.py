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


def test_tutorial_linear_and_bifurcation_runs_with_canonical_api():
    tutorial = _load_tutorial_module()

    linear = tutorial.run_linear_tutorial()
    bifurcation = tutorial.run_bifurcation_tutorial()

    assert linear["uns"]["simulation_config"]["capture_model"] == "poisson_capture"
    assert linear["uns"]["simulation_config"]["alpha_source_mode"] == "continuous_program"
    assert bifurcation["uns"]["simulation_config"]["capture_model"] == "binomial_capture"
    assert bifurcation["uns"]["simulation_config"]["alpha_source_mode"] == "state_anchor"
    assert bifurcation["uns"]["simulation_config"]["trunk_state"] == "root"
    assert bifurcation["uns"]["simulation_config"]["transition_schedule"] == "sigmoid"
    assert linear["layers"]["true_spliced"].shape[0] == tutorial.linear_parameters()["n_cells"]
    assert set(bifurcation["obs"]["branch"].unique()) == {"trunk", "branch_0", "branch_1"}


def test_public_api_does_not_export_legacy_interfaces():
    import nvsim

    assert "simulate_bifurcation_legacy" not in nvsim.__all__
    assert "calibrate_grn_thresholds" not in nvsim.__all__
