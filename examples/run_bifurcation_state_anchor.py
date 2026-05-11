"""Run a bifurcation trajectory with SERGIO-style state-anchor alpha sources."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from run_mvp_linear import build_example_grn

from nvsim.output import to_anndata
from nvsim.production import StateProductionProfile
from nvsim.simulate import simulate_bifurcation


def _plot_master_alpha(result: dict, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped alpha diagnostic plot")
        return

    obs = result["obs"]
    var = result["var"]
    alpha = result["layers"]["true_alpha"]
    master_genes = list(var.index[var["gene_role"] == "master_regulator"])
    gene_to_idx = {gene: i for i, gene in enumerate(var.index)}
    fig, axes = plt.subplots(len(master_genes), 1, figsize=(7, 2.2 * len(master_genes)), sharex=True)
    if len(master_genes) == 1:
        axes = [axes]
    for ax, gene in zip(axes, master_genes):
        idx = gene_to_idx[gene]
        for branch, frame in obs.groupby("branch", sort=False):
            rows = frame.index.map(lambda name: obs.index.get_loc(name)).to_numpy()
            ax.scatter(frame["pseudotime"], alpha[rows, idx], s=10, label=branch, alpha=0.75)
        ax.set_ylabel(f"{gene} alpha")
    axes[-1].set_xlabel("pseudotime")
    axes[0].legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / "master_alpha_by_pseudotime.png", dpi=180)
    plt.close(fig)


def main() -> None:
    grn = build_example_grn()
    profile = StateProductionProfile(
        pd.DataFrame(
            {
                "g0": [0.4, 1.4, 0.15],
                "g1": [0.8, 0.35, 1.25],
                "g2": [0.6, 1.1, 0.2],
            },
            index=["progenitor", "lineage_A", "lineage_B"],
        )
    )
    result = simulate_bifurcation(
        grn,
        n_trunk_cells=50,
        n_branch_cells={"branch_0": 60, "branch_1": 60},
        trunk_time=2.0,
        branch_time=2.5,
        dt=0.02,
        alpha_source_mode="state_anchor",
        production_profile=profile,
        trunk_state="progenitor",
        branch_child_states={"branch_0": "lineage_A", "branch_1": "lineage_B"},
        transition_schedule="sigmoid",
        transition_midpoint=0.5,
        transition_steepness=10.0,
        seed=123,
        capture_rate=0.5,
        dropout_rate=0.02,
    )

    out_dir = Path(__file__).with_name("outputs") / "bifurcation_state_anchor"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        to_anndata(result).write_h5ad(out_dir / "bifurcation_state_anchor.h5ad")
    except ImportError:
        print("anndata is not installed; skipped h5ad export")
    _plot_master_alpha(result, out_dir)
    print({
        "alpha_source_mode": result["uns"]["simulation_config"]["alpha_source_mode"],
        "branch_child_states": result["uns"]["simulation_config"]["branch_child_states"],
        "transition_schedule": result["uns"]["simulation_config"]["transition_schedule"],
        "output_dir": str(out_dir),
    })


if __name__ == "__main__":
    main()
