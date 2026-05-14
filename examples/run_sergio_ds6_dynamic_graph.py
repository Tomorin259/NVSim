#!/usr/bin/env python3
"""Run the SERGIO DS6 dynamic benchmark with NVSim's SERGIO-style mode."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nvsim.modes import StateGraph
from nvsim.output import to_anndata
from nvsim.plotting import plot_phase_gallery, plot_showcase
from nvsim.production import StateProductionProfile
from nvsim.sergio_io import load_sergio_targets_regs
from nvsim.simulate import simulate


def _json_safe(value):
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value

DEFAULT_DATASET_DIR = Path(
    "/mnt/second19T/zhaozelin/simulator/SERGIO/data_sets/De-noised_100G_6T_300cPerT_dynamics_7_DS6"
)
DEFAULT_OUTPUT_DIR = ROOT / "examples" / "outputs" / "sergio_100g_ds6_sergio_differentiation_20260514"


def load_ds6_inputs(dataset_dir: Path = DEFAULT_DATASET_DIR):
    interaction = dataset_dir / "Interaction_cID_7.txt"
    regs = dataset_dir / "Regs_cID_7.txt"
    bmat = dataset_dir / "bMat_cID7.tab"
    for path in [interaction, regs, bmat]:
        if not path.exists():
            raise FileNotFoundError(f"missing SERGIO DS6 file: {path}")
    inputs = load_sergio_targets_regs(interaction, regs, shared_coop_state=2)
    return inputs, bmat


def load_bifurcation_matrix(path: Path) -> pd.DataFrame:
    matrix = pd.read_csv(path, sep="\t", header=None, dtype=float)
    n_states = int(matrix.shape[0])
    if matrix.shape[1] != n_states:
        raise ValueError(f"bifurcation matrix must be square, got {matrix.shape}")
    states = [f"bin_{idx}" for idx in range(n_states)]
    matrix.index = states
    matrix.columns = states
    return matrix


def graph_from_bifurcation_matrix(matrix: pd.DataFrame) -> StateGraph:
    edges: list[dict[str, str]] = []
    for parent in matrix.index:
        for child in matrix.columns:
            if float(matrix.loc[parent, child]) > 0:
                edges.append({"parent_state": str(parent), "child_state": str(child)})
    if not edges:
        raise ValueError("bifurcation matrix did not contain any directed edges")
    return StateGraph(pd.DataFrame.from_records(edges), states=tuple(str(state) for state in matrix.index))


def build_result(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, object]:
    inputs, bmat_path = load_ds6_inputs()
    bmat = load_bifurcation_matrix(bmat_path)
    graph = graph_from_bifurcation_matrix(bmat)
    profile = StateProductionProfile(inputs.master_production)
    states = [str(state) for state in profile.states]
    n_cells_per_state = {state: 300 for state in states}
    state_time = {state: 3.0 for state in states}
    result = simulate(
        inputs.grn,
        simulation_mode="sergio_differentiation",
        graph=graph,
        production_profile=profile,
        master_regulators=inputs.master_regulators,
        n_cells_per_state=n_cells_per_state,
        root_time=3.0,
        state_time=state_time,
        dt=0.01,
        capture_rate=1.0,
        poisson_observed=False,
        dropout_rate=0.0,
        seed=7,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "result": result,
        "output_dir": output_dir,
        "graph": graph,
        "bmat": bmat,
        "dataset_dir": str(DEFAULT_DATASET_DIR),
    }


def save_outputs(bundle: dict[str, object]) -> dict[str, object]:
    result = bundle["result"]
    output_dir = Path(bundle["output_dir"])
    graph = bundle["graph"]
    bmat = bundle["bmat"]

    adata = to_anndata(result)
    h5ad_path = output_dir / "sergio_100g_ds6_sergio_differentiation.h5ad"
    adata.write_h5ad(h5ad_path)

    obs_path = output_dir / "obs.csv"
    var_path = output_dir / "var.csv"
    edges_path = output_dir / "differentiation_graph_edges.csv"
    bmat_path = output_dir / "bifurcation_matrix.csv"
    layers_path = output_dir / "layers.npz"

    result["obs"].to_csv(obs_path, index=False)
    result["var"].to_csv(var_path, index=False)
    graph.edges.to_csv(edges_path, index=False)
    bmat.to_csv(bmat_path)
    np.savez_compressed(layers_path, **{name: np.asarray(value) for name, value in result["layers"].items()})

    showcase = plot_showcase(
        result,
        output_dir / "velocity_showcase",
        expression_layer="true",
        basis="umap",
        n_pcs=30,
        n_neighbors=30,
        random_state=7,
    )
    phase_genes = [str(gene) for gene in adata.var_names[:100]]
    phase_gallery_path = output_dir / "phase_gallery_all_100_genes_true.png"
    plot_phase_gallery(result, genes=phase_genes, mode="true", output_path=phase_gallery_path)

    summary = {
        "dataset_dir": bundle["dataset_dir"],
        "dataset_name": DEFAULT_DATASET_DIR.name,
        "mode": "nvsim_sergio_differentiation",
        "output_dir": str(output_dir),
        "h5ad": str(h5ad_path),
        "obs_csv": str(obs_path),
        "var_csv": str(var_path),
        "graph_csv": str(edges_path),
        "bifurcation_matrix_csv": str(bmat_path),
        "layers_npz": str(layers_path),
        "plots": {
            "phase_gallery": str(phase_gallery_path),
            "velocity_showcase": showcase["files"],
        },
        "state_counts": result["obs"]["state"].astype(str).value_counts().sort_index().to_dict(),
        "simulation_config": _json_safe(result["uns"]["simulation_config"]),
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    bundle = build_result()
    summary = save_outputs(bundle)
    print(json.dumps({
        "output_dir": summary["output_dir"],
        "h5ad": summary["h5ad"],
        "states": summary["state_counts"],
        "simulator": summary["simulation_config"]["simulator"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
