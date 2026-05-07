"""Plot the SERGIO 100-gene DS8 NVSim bifurcation example."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from nvsim.grn import calibrate_half_response
from nvsim.plotting import (
    compute_umap_embedding,
    plot_embedding_by_branch,
    plot_gene_dynamics_over_pseudotime,
    plot_overview_panel,
    plot_phase_portrait,
    plot_selected_gene_panel,
    select_representative_genes_by_dynamics,
)
from nvsim.sergio_io import load_sergio_targets_regs

from run_sergio_100g_ds8_bifurcation import (
    BMAT_FILE,
    BRANCH_STATES,
    DATASET_DIR,
    OUTPUT_DIR,
    OUTPUT_H5AD,
    REGS_FILE,
    SUMMARY_JSON,
    TARGETS_FILE,
    TRUNK_STATE,
    build_sergio_100g_ds8_bifurcation_result,
    load_sergio_noiseless_mean_expression,
)


OUTPUT_BASE = OUTPUT_DIR / "plots"


def _save_observed_phase(result, gene: str, label: str, output_dir: Path, suffix: str) -> Path | None:
    path = output_dir / f"phase_portrait_{label}_{gene}_{suffix}.png"
    try:
        plot_phase_portrait(result, gene, mode="observed", output_path=path)
    except KeyError as exc:
        print(f"skipping {suffix} phase portrait for {gene}: {exc}")
        return None
    return path


def _load_grn():
    sergio_inputs = load_sergio_targets_regs(TARGETS_FILE, REGS_FILE)
    mean_expression = load_sergio_noiseless_mean_expression(DATASET_DIR)
    return calibrate_half_response(sergio_inputs.grn, mean_expression)


def _load_or_build_noisy():
    if OUTPUT_H5AD.exists():
        try:
            import anndata as ad
        except ImportError:
            pass
        else:
            print(f"loading existing {OUTPUT_H5AD}")
            return ad.read_h5ad(OUTPUT_H5AD)
    noisy, _, _ = build_sergio_100g_ds8_bifurcation_result(
        capture_rate=0.6,
        dropout_rate=0.01,
        poisson_observed=True,
    )
    return noisy


def main() -> None:
    base_dir = OUTPUT_BASE
    if base_dir.exists():
        shutil.rmtree(base_dir)
    overview_dir = base_dir / "overview"
    true_dir = base_dir / "true"
    observed_dir = base_dir / "observed"
    lownoise_dir = base_dir / "observed_lownoise"
    genes_dir = base_dir / "genes"
    diagnostics_dir = base_dir / "diagnostics"
    for directory in (overview_dir, true_dir, observed_dir, lownoise_dir, genes_dir, diagnostics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    grn = _load_grn()
    noisy = _load_or_build_noisy()
    lownoise, _, _ = build_sergio_100g_ds8_bifurcation_result(
        capture_rate=1.0,
        dropout_rate=0.0,
        poisson_observed=False,
    )
    selection = select_representative_genes_by_dynamics(noisy, grn)
    selected = selection["genes"]
    generated: list[Path] = []

    generated.append(overview_dir / "trajectory_overview_panel.png")
    plot_overview_panel(noisy, lownoise=lownoise, random_state=2088, output_path=generated[-1])
    generated.append(overview_dir / "selected_genes_panel.png")
    plot_selected_gene_panel(noisy, selected, include_velocity_u=True, output_path=generated[-1])

    generated.append(true_dir / "embedding_pca_true_by_branch.png")
    plot_embedding_by_branch(noisy, method="pca", layer_preference="true", output_path=generated[-1])
    for label, gene in selected.items():
        generated.append(genes_dir / f"phase_portrait_{label}_{gene}_true.png")
        plot_phase_portrait(noisy, gene, mode="true", connect_by_pseudotime=True, output_path=generated[-1])
        generated.append(genes_dir / f"gene_dynamics_{label}_{gene}.png")
        plot_gene_dynamics_over_pseudotime(noisy, gene, include_velocity_u=True, output_path=generated[-1])

    plt.close("all")

    generated.append(observed_dir / "embedding_pca_observed_by_branch.png")
    plot_embedding_by_branch(noisy, method="pca", layer_preference="observed", output_path=generated[-1])
    _, umap_method, _ = compute_umap_embedding(noisy, random_state=2088, layer_preference="observed")
    umap_available = umap_method == "umap"
    if umap_available:
        generated.append(observed_dir / "embedding_umap_observed_by_branch.png")
        plot_embedding_by_branch(
            noisy, method="umap", layer_preference="observed", random_state=2088, output_path=generated[-1]
        )

    plt.close("all")

    generated.append(lownoise_dir / "embedding_pca_observed_lownoise_by_branch.png")
    plot_embedding_by_branch(lownoise, method="pca", layer_preference="observed", output_path=generated[-1])
    for label, gene in selected.items():
        path = _save_observed_phase(lownoise, gene, label, genes_dir, "observed_lownoise")
        if path is not None:
            generated.append(path)

    selected_lines = [
        f"Source targets file: {TARGETS_FILE}",
        f"Source regs file: {REGS_FILE}",
        f"Source bMat file: {BMAT_FILE}",
        f"Summary json: {SUMMARY_JSON}",
        f"n_genes: {len(grn.genes)}",
        f"n_edges: {grn.edges.shape[0]}",
        f"trunk_state: {TRUNK_STATE}",
        f"branch_states: {BRANCH_STATES}",
        "",
    ]
    for label, gene in selected.items():
        selected_lines.append(f"{label}: {gene}")
        selected_lines.append(f"  post_branch_alpha_difference: {selection['alpha_differences'][label]:.6g}")
        edges = selection["edges"][label]
        if edges.empty:
            selected_lines.append("  incoming_edges: none")
        else:
            selected_lines.append("  incoming_edges:")
            for _, row in edges.iterrows():
                selected_lines.append(
                    "    regulator={reg} target={target} sign={sign} weight={weight:.4g} half_response={half:.4g}".format(
                        reg=row["regulator"],
                        target=row["target"],
                        sign=row["sign"],
                        weight=row["weight"],
                        half=row["half_response"],
                    )
                )
    (diagnostics_dir / "selected_genes.txt").write_text("\n".join(selected_lines) + "\n", encoding="utf-8")
    (diagnostics_dir / "README.txt").write_text(
        "SERGIO 100-gene DS8 targets/regs quick-look plots routed through NVSim deterministic ODE. "
        "True plots remain the primary scientific validation view; observed plots show technical-noise effects; "
        "observed_lownoise uses poisson_observed=False for visualization/debugging only.\n",
        encoding="utf-8",
    )

    plt.close("all")
    print(f"umap_available={umap_available}")
    print(f"selected_genes={selected}")
    print(f"saved plots to {base_dir}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
