"""Plot comparisons for regulator-activity benchmark on SERGIO 100G DS8."""

from __future__ import annotations

from pathlib import Path
import json
import shutil
import sys

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nvsim.grn import calibrate_half_response
from nvsim.output import to_anndata
from nvsim.plotting import (
    plot_embedding_by_branch,
    plot_embedding_with_velocity,
    plot_phase_portrait,
    plot_phase_portrait_gallery,
    select_representative_genes_by_dynamics,
)
from nvsim.sergio_io import load_sergio_targets_regs

from run_sergio_100g_ds8_regulator_activity_benchmark import (
    DATASET_DIR,
    MODES,
    OUTPUT_ROOT,
    REGS_FILE,
    TARGETS_FILE,
    build_benchmark_result,
    load_sergio_noiseless_mean_expression,
)


PLOTS_DIR = OUTPUT_ROOT / "plots"


def _mode_h5ad(mode: str) -> Path:
    return OUTPUT_ROOT / mode / f"sergio_100g_ds8_{mode}.h5ad"


def _load_grn():
    sergio_inputs = load_sergio_targets_regs(TARGETS_FILE, REGS_FILE)
    mean_expression = load_sergio_noiseless_mean_expression(DATASET_DIR)
    return calibrate_half_response(sergio_inputs.grn, mean_expression)


def _load_or_build(mode: str):
    path = _mode_h5ad(mode)
    if path.exists():
        return ad.read_h5ad(path)
    result, _, _ = build_benchmark_result(mode)
    return to_anndata(result)


def _comparison_grid(plot_fn, datasets: dict[str, ad.AnnData], output_path: Path, **kwargs) -> None:
    fig, axes = plt.subplots(1, len(MODES), figsize=(5 * len(MODES), 4.5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, mode in zip(axes, MODES):
        plot_fn(datasets[mode], ax=ax, **kwargs)
        ax.set_title(f"{mode}")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _selected_phase_compare(datasets: dict[str, ad.AnnData], selected: dict[str, str], output_path: Path) -> None:
    n_rows = len(selected)
    n_cols = len(MODES)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.6 * n_cols, 3.2 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )
    for row_idx, (label, gene) in enumerate(selected.items()):
        for col_idx, mode in enumerate(MODES):
            ax = axes[row_idx, col_idx]
            plot_phase_portrait(datasets[mode], gene, mode="true", connect_by_pseudotime=True, ax=ax)
            if row_idx == 0:
                ax.set_title(f"{mode}")
        axes[row_idx, 0].annotate(
            f"{label}: {gene}",
            xy=(0, 0.5),
            xycoords="axes fraction",
            xytext=(-axes[row_idx, 0].yaxis.labelpad - 10, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            rotation=90,
            fontsize=10,
        )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _selected_dynamics_compare(datasets: dict[str, ad.AnnData], selected: dict[str, str], output_path: Path) -> None:
    quantities = [
        ("true_alpha", "alpha"),
        ("true_unspliced", "u"),
        ("true_spliced", "s"),
        ("true_velocity", "v"),
        ("true_velocity_u", "du/dt"),
    ]
    branches_order = ("trunk", "branch_0", "branch_1")
    branch_colors = {"trunk": "tab:blue", "branch_0": "tab:orange", "branch_1": "tab:green"}
    for label, gene in selected.items():
        fig, axes = plt.subplots(
            len(quantities),
            len(MODES),
            figsize=(4.6 * len(MODES), 2.0 * len(quantities) + 0.4),
            sharex=False,
            constrained_layout=True,
            squeeze=False,
        )
        for col_idx, mode in enumerate(MODES):
            adata = datasets[mode]
            gene_idx = list(map(str, adata.var_names)).index(str(gene))
            pt = adata.obs["pseudotime"].to_numpy(dtype=float)
            branches = adata.obs["branch"].astype(str).to_numpy()
            for row_idx, (layer_name, ylabel) in enumerate(quantities):
                ax = axes[row_idx, col_idx]
                values = np.asarray(adata.layers[layer_name], dtype=float)[:, gene_idx]
                for branch in branches_order:
                    mask = branches == branch
                    if not mask.any():
                        continue
                    order = np.argsort(pt[mask])
                    ax.plot(
                        pt[mask][order],
                        values[mask][order],
                        marker="o",
                        markersize=2.0,
                        linewidth=1.0,
                        color=branch_colors.get(branch, "0.3"),
                        label=branch,
                    )
                if row_idx == 0:
                    ax.set_title(mode)
                if col_idx == 0:
                    ax.set_ylabel(ylabel)
                if row_idx == len(quantities) - 1:
                    ax.set_xlabel("pseudotime")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            axes[0, 0].legend(handles, labels, frameon=False, fontsize=8)
        fig.suptitle(f"Gene dynamics comparison: {label} ({gene})", fontsize=12)
        fig.savefig(output_path.parent / f"compare_gene_dynamics_{label}_{gene}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def _save_selected_genes(selected: dict[str, str], output_path: Path) -> None:
    lines = [f"{label}: {gene}" for label, gene in selected.items()]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if PLOTS_DIR.exists():
        shutil.rmtree(PLOTS_DIR)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    grn = _load_grn()
    datasets = {mode: _load_or_build(mode) for mode in MODES}
    selection = select_representative_genes_by_dynamics(datasets["spliced"], grn)
    selected = selection["genes"]

    _comparison_grid(
        plot_embedding_by_branch,
        datasets,
        PLOTS_DIR / "compare_true_pca_by_branch.png",
        method="pca",
        layer_preference="true",
    )
    _comparison_grid(
        plot_embedding_with_velocity,
        datasets,
        PLOTS_DIR / "compare_true_pca_with_velocity.png",
        method="pca",
        layer_preference="true",
    )
    _comparison_grid(
        plot_embedding_by_branch,
        datasets,
        PLOTS_DIR / "compare_observed_pca_by_branch.png",
        method="pca",
        layer_preference="observed",
    )

    for mode, adata in datasets.items():
        plot_phase_portrait_gallery(
            adata,
            mode="true",
            connect_by_pseudotime=True,
            max_cols=5,
            panel_size=2.0,
            output_path=PLOTS_DIR / f"all_genes_phase_portraits_true_{mode}.png",
        )

    _selected_phase_compare(datasets, selected, PLOTS_DIR / "compare_selected_gene_phase_portraits_true.png")
    _selected_dynamics_compare(datasets, selected, PLOTS_DIR / "compare_selected_gene_dynamics.png")
    _save_selected_genes(selected, PLOTS_DIR / "selected_genes.txt")

    comparison_metrics = OUTPUT_ROOT / "comparison_metrics.json"
    if comparison_metrics.exists():
        payload = json.loads(comparison_metrics.read_text(encoding="utf-8"))
        (PLOTS_DIR / "comparison_metrics_snapshot.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(f"selected_genes={selected}")
    print(f"saved plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
