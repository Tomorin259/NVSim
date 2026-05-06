"""Generate organized quick-look plots for the bifurcation NVSim MVP."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from run_mvp_bifurcation import build_bifurcation_result
from run_mvp_linear import build_example_grn

from nvsim.plotting import (
    compute_umap_embedding,
    plot_embedding_by_branch,
    plot_embedding_by_pseudotime,
    plot_embedding_with_velocity,
    plot_gene_dynamics_over_pseudotime,
    plot_phase_portrait,
    select_representative_genes_by_dynamics,
)


def _save_observed_phase(result, gene: str, label: str, output_dir: Path, suffix: str) -> Path | None:
    path = output_dir / f"phase_portrait_{label}_{gene}_{suffix}.png"
    try:
        plot_phase_portrait(result, gene, mode="observed", output_path=path)
    except KeyError as exc:
        print(f"skipping {suffix} phase portrait for {gene}: {exc}")
        return None
    return path


def main() -> None:
    base_dir = Path(__file__).with_name("outputs") / "bifurcation_20gene_3master" / "plots"
    if base_dir.exists():
        shutil.rmtree(base_dir)
    true_dir = base_dir / "true"
    observed_dir = base_dir / "observed"
    lownoise_dir = base_dir / "observed_lownoise"
    diagnostics_dir = base_dir / "diagnostics"
    for directory in (true_dir, observed_dir, lownoise_dir, diagnostics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    grn = build_example_grn()
    noisy = build_bifurcation_result(capture_rate=0.5, dropout_rate=0.02)
    lownoise = build_bifurcation_result(capture_rate=1.0, dropout_rate=0.0, poisson_observed=False)
    selection = select_representative_genes_by_dynamics(noisy, grn)
    selected = selection["genes"]
    generated: list[Path] = []

    generated.append(true_dir / "embedding_pca_true_by_pseudotime.png")
    plot_embedding_by_pseudotime(noisy, method="pca", layer_preference="true", output_path=generated[-1])
    generated.append(true_dir / "embedding_pca_true_by_branch.png")
    plot_embedding_by_branch(noisy, method="pca", layer_preference="true", output_path=generated[-1])
    generated.append(true_dir / "embedding_pca_true_with_velocity.png")
    plot_embedding_with_velocity(noisy, method="pca", layer_preference="true", output_path=generated[-1])
    for label, gene in selected.items():
        generated.append(true_dir / f"phase_portrait_{label}_{gene}_true.png")
        plot_phase_portrait(noisy, gene, mode="true", output_path=generated[-1])
        generated.append(true_dir / f"gene_dynamics_{label}_{gene}.png")
        plot_gene_dynamics_over_pseudotime(noisy, gene, output_path=generated[-1])

    plt.close("all")

    generated.append(observed_dir / "embedding_pca_observed_by_pseudotime.png")
    plot_embedding_by_pseudotime(noisy, method="pca", layer_preference="observed", output_path=generated[-1])
    generated.append(observed_dir / "embedding_pca_observed_by_branch.png")
    plot_embedding_by_branch(noisy, method="pca", layer_preference="observed", output_path=generated[-1])
    _, umap_method, _ = compute_umap_embedding(noisy, random_state=123, layer_preference="observed")
    umap_available = umap_method == "umap"
    if umap_available:
        generated.append(observed_dir / "embedding_umap_observed_by_pseudotime.png")
        plot_embedding_by_pseudotime(noisy, method="umap", layer_preference="observed", random_state=123, output_path=generated[-1])
        generated.append(observed_dir / "embedding_umap_observed_by_branch.png")
        plot_embedding_by_branch(noisy, method="umap", layer_preference="observed", random_state=123, output_path=generated[-1])
    else:
        print("umap-learn is unavailable; skipping observed UMAP plots")
    for label, gene in selected.items():
        path = _save_observed_phase(noisy, gene, label, observed_dir, "observed")
        if path is not None:
            generated.append(path)

    plt.close("all")

    generated.append(lownoise_dir / "embedding_pca_observed_lownoise_by_pseudotime.png")
    plot_embedding_by_pseudotime(lownoise, method="pca", layer_preference="observed", output_path=generated[-1])
    generated.append(lownoise_dir / "embedding_pca_observed_lownoise_by_branch.png")
    plot_embedding_by_branch(lownoise, method="pca", layer_preference="observed", output_path=generated[-1])
    for label, gene in selected.items():
        path = _save_observed_phase(lownoise, gene, label, lownoise_dir, "observed_lownoise")
        if path is not None:
            generated.append(path)

    (diagnostics_dir / "README.txt").write_text(
        "Bifurcation quick-look guide:\n"
        "Start with true/embedding_pca_true_by_branch.png and true/embedding_pca_true_with_velocity.png.\n"
        "The trunk is simulated first; branch_0 and branch_1 inherit the terminal trunk u/s state, then evolve independently.\n"
        "Observed plots are technical-noise diagnostics. observed_lownoise uses poisson_observed=False for visualization/debugging.\n"
        "UMAP can fragment sparse toy data and should not be overinterpreted.\n"
        "PCA velocity arrows are qualitative diagnostics, not a full scVelo-style velocity embedding.\n",
        encoding="utf-8",
    )

    selected_lines = []
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
                    f"    regulator={row['regulator']} target={row['target']} sign={row['sign']} weight={row['weight']}"
                )
    (diagnostics_dir / "selected_genes.txt").write_text("\n".join(selected_lines) + "\n", encoding="utf-8")

    plt.close("all")
    print(f"umap_available={umap_available}")
    print(f"selected_genes={selected}")
    print(f"saved plots to {base_dir}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
