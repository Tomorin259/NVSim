#!/usr/bin/env python3
"""Run the DS6 graph baseline with corrected step transitions and diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

try:
    import scvelo as scv
except Exception:
    scv = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nvsim.modes import StateGraph
from nvsim.output import to_anndata
from nvsim.plotting import plot_gene_dynamics, plot_phase_gallery, plot_showcase
from nvsim.production import StateProductionProfile
from nvsim.sergio_io import load_sergio_targets_regs
from nvsim.simulate import simulate

DEFAULT_DATASET_DIR = Path(
    "/mnt/second19T/zhaozelin/simulator/SERGIO/data_sets/De-noised_100G_6T_300cPerT_dynamics_7_DS6"
)
DEFAULT_OUTPUT_DIR = ROOT / "examples" / "outputs" / "ds6_pt_s3_c300_stepfix"
REGULATION_PAIRS = [("9", "70", "bin_3"), ("5", "87", "bin_4"), ("79", "56", "bin_5")]
PHASE_GENES = ["9", "70", "5", "87", "79", "56"]


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
        graph=graph,
        production_profile=profile,
        master_regulators=inputs.master_regulators,
        alpha_source_mode="state_anchor",
        child_initialization_policy="parent_terminal",
        sampling_policy="state_transient",
        half_response_calibration="auto",
        regulator_activity="unspliced",
        transition_schedule="step",
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


def save_core_outputs(bundle: dict[str, object]) -> dict[str, object]:
    result = bundle["result"]
    output_dir = Path(bundle["output_dir"])
    graph = bundle["graph"]
    bmat = bundle["bmat"]

    adata = to_anndata(result)
    h5ad_path = output_dir / "ds6_stepfix_clean_simulation.h5ad"
    adata.write_h5ad(h5ad_path)

    obs_path = output_dir / "obs.csv"
    var_path = output_dir / "var.csv"
    edges_path = output_dir / "graph_edges.csv"
    bmat_out_path = output_dir / "bifurcation_matrix.csv"
    layers_path = output_dir / "layers.npz"

    result["obs"].to_csv(obs_path, index=False)
    result["var"].to_csv(var_path)
    graph.edges.to_csv(edges_path, index=False)
    bmat.to_csv(bmat_out_path)
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

    phase_gallery_path = output_dir / "phase_gallery_all_100_genes_true.png"
    plot_phase_gallery(result, genes=[str(gene) for gene in adata.var_names[:100]], mode="true", output_path=phase_gallery_path)

    return {
        "adata": adata,
        "h5ad": h5ad_path,
        "obs_csv": obs_path,
        "var_csv": var_path,
        "graph_csv": edges_path,
        "bifurcation_matrix_csv": bmat_out_path,
        "layers_npz": layers_path,
        "velocity_showcase": showcase["files"],
        "phase_gallery": phase_gallery_path,
    }


def save_official_style_outputs(adata: ad.AnnData, output_dir: Path) -> dict[str, object]:
    off = output_dir / "official_style_scanpy_scvelo"
    off.mkdir(parents=True, exist_ok=True)

    clean = adata.copy()
    clean.layers["total"] = np.asarray(clean.layers["true_unspliced"]) + np.asarray(clean.layers["true_spliced"])
    clean.X = clean.layers["total"].copy()
    sc.pp.normalize_total(clean, target_sum=1e4)
    sc.pp.log1p(clean)
    sc.pp.pca(clean, n_comps=min(30, clean.n_vars - 1))
    sc.pp.neighbors(clean, n_neighbors=30, n_pcs=min(30, clean.obsm["X_pca"].shape[1]))
    sc.tl.umap(clean, random_state=7)
    clean_h5ad = off / "ds6_stepfix_clean_total_scanpy.h5ad"
    clean.write_h5ad(clean_h5ad)

    plots = []
    for color, fn in [
        ("state", "scanpy_umap_cell_type_clean_total.png"),
        ("branch", "scanpy_umap_branch_clean_total.png"),
        ("pseudotime", "scanpy_umap_pseudotime_clean_total.png"),
    ]:
        fig = sc.pl.umap(clean, color=color, show=False, return_fig=True)
        path = off / fn
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        plots.append(path)

    scvelo_ok = False
    if scv is not None:
        vel = adata.copy()
        vel.X = np.asarray(vel.layers["true_spliced"]).copy()
        vel.layers["spliced"] = np.asarray(vel.layers["true_spliced"]).copy()
        vel.layers["unspliced"] = np.asarray(vel.layers["true_unspliced"]).copy()
        vel.layers["velocity"] = np.asarray(vel.layers["true_velocity"]).copy()
        vel.obsm["X_umap"] = clean.obsm["X_umap"].copy()
        sc.pp.neighbors(vel, n_neighbors=30, use_rep="X")
        try:
            scv.tl.velocity_graph(vel, vkey="velocity")
            fig = scv.pl.umap(vel, color="state", show=False, return_fig=True)
            path = off / "scvelo_umap_cell_type.png"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            plots.append(path)
            scvelo_ok = True
        except Exception:
            scvelo_ok = False

    summary = {
        "clean_total_h5ad": str(clean_h5ad),
        "plots": [str(path) for path in plots],
        "scvelo_velocity_graph_ok": scvelo_ok,
    }
    (off / "official_style_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def save_regulation_diagnostics(adata: ad.AnnData, output_dir: Path) -> dict[str, object]:
    diag = output_dir / "regulation_diagnostics"
    diag.mkdir(parents=True, exist_ok=True)

    adata.layers["true_total"] = np.asarray(adata.layers["true_unspliced"]) + np.asarray(adata.layers["true_spliced"])
    plot_genes = [gene for pair in REGULATION_PAIRS for gene in pair[:2]]
    plot_gene_dynamics(
        adata,
        plot_genes,
        output_path=diag / "gene_dynamics_regulators_targets.png",
        quantities=("true_alpha", "true_unspliced", "true_spliced", "true_total"),
    )
    plot_phase_gallery(
        adata,
        genes=PHASE_GENES,
        mode="true",
        output_path=diag / "phase_gallery_regulators_targets_true_velocity.png",
    )

    var_names = list(map(str, adata.var_names))
    rows = []
    summary_pairs = []
    states = sorted(pd.unique(adata.obs["state"].astype(str)))
    for reg, tgt, leaf in REGULATION_PAIRS:
        ridx = var_names.index(reg)
        tidx = var_names.index(tgt)
        state_stats = {}
        for state in states:
            mask = (adata.obs["state"].astype(str) == state).to_numpy()
            reg_alpha = float(np.asarray(adata.layers["true_alpha"])[:, ridx][mask].mean())
            reg_u = float(np.asarray(adata.layers["true_unspliced"])[:, ridx][mask].mean())
            reg_s = float(np.asarray(adata.layers["true_spliced"])[:, ridx][mask].mean())
            reg_total = reg_u + reg_s
            tgt_alpha = float(np.asarray(adata.layers["true_alpha"])[:, tidx][mask].mean())
            tgt_u = float(np.asarray(adata.layers["true_unspliced"])[:, tidx][mask].mean())
            tgt_s = float(np.asarray(adata.layers["true_spliced"])[:, tidx][mask].mean())
            tgt_total = tgt_u + tgt_s
            state_stats[state] = {
                "reg_alpha": reg_alpha,
                "reg_u": reg_u,
                "reg_s": reg_s,
                "reg_total": reg_total,
                "tgt_alpha": tgt_alpha,
                "tgt_u": tgt_u,
                "tgt_s": tgt_s,
                "tgt_total": tgt_total,
            }
            rows.append(
                {
                    "pair": f"{reg}->{tgt}",
                    "state": state,
                    "regulator": reg,
                    "target": tgt,
                    "reg_alpha": reg_alpha,
                    "reg_u": reg_u,
                    "reg_s": reg_s,
                    "reg_total": reg_total,
                    "target_alpha": tgt_alpha,
                    "target_u": tgt_u,
                    "target_s": tgt_s,
                    "target_total": tgt_total,
                }
            )
        other_leaf_totals = [state_stats[s]["tgt_total"] for s in ["bin_3", "bin_4", "bin_5"] if s != leaf]
        summary_pairs.append(
            {
                "regulator": reg,
                "target": tgt,
                "expected_leaf": leaf,
                "states": state_stats,
                "target_leaf_margin_total": float(state_stats[leaf]["tgt_total"] - max(other_leaf_totals)),
            }
        )

    means_csv = diag / "gene_state_means_regulators_targets.csv"
    summary_json = diag / "regulation_pair_summary.json"
    pd.DataFrame(rows).to_csv(means_csv, index=False)
    summary_json.write_text(json.dumps({"pairs": summary_pairs}, indent=2) + "\n")
    (diag / "README.txt").write_text(
        "Representative direct-edge diagnostics for 9->70 (bin_3), 5->87 (bin_4), and 79->56 (bin_5).\n"
    )
    return {
        "plot": str(diag / "gene_dynamics_regulators_targets.png"),
        "phase_gallery": str(diag / "phase_gallery_regulators_targets_true_velocity.png"),
        "means_csv": str(means_csv),
        "summary_json": str(summary_json),
    }


def save_outputs(bundle: dict[str, object]) -> dict[str, object]:
    output_dir = Path(bundle["output_dir"])
    core = save_core_outputs(bundle)
    adata = core["adata"]
    official = save_official_style_outputs(adata, output_dir)
    diagnostics = save_regulation_diagnostics(adata, output_dir)

    summary = {
        "dataset_dir": bundle["dataset_dir"],
        "dataset_name": DEFAULT_DATASET_DIR.name,
        "mode": "nvsim_dynamic_graph_stepfix",
        "output_dir": str(output_dir),
        "h5ad": str(core["h5ad"]),
        "obs_csv": str(core["obs_csv"]),
        "var_csv": str(core["var_csv"]),
        "graph_csv": str(core["graph_csv"]),
        "bifurcation_matrix_csv": str(core["bifurcation_matrix_csv"]),
        "layers_npz": str(core["layers_npz"]),
        "plots": {
            "phase_gallery": str(core["phase_gallery"]),
            "velocity_showcase": core["velocity_showcase"],
            "official_style": official,
            "regulation_diagnostics": diagnostics,
        },
        "state_counts": bundle["result"]["obs"]["state"].astype(str).value_counts().sort_index().to_dict(),
        "simulation_config": _json_safe(bundle["result"]["uns"]["simulation_config"]),
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    bundle = build_result()
    summary = save_outputs(bundle)
    print(
        json.dumps(
            {
                "output_dir": summary["output_dir"],
                "h5ad": summary["h5ad"],
                "states": summary["state_counts"],
                "simulator": summary["simulation_config"]["simulator"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
