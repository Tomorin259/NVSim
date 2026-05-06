"""Generate organized quick-look plots for the linear NVSim MVP."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from run_mvp_linear import build_example_grn

from nvsim.plotting import (
    compute_umap_embedding,
    plot_embedding_by_pseudotime,
    plot_embedding_with_velocity,
    plot_gene_dynamics_over_pseudotime,
    plot_phase_portrait,
)
from nvsim.programs import linear_increase, sigmoid_decrease
from nvsim.simulate import simulate_linear


def _safe_gene(grn, preferred: str, fallback_index: int) -> str:
    if preferred in grn.genes:
        return preferred
    return grn.genes[min(fallback_index, len(grn.genes) - 1)]


def select_representative_genes(grn) -> dict[str, str]:
    """Pick one master regulator, one activation target, and one repression target."""

    incoming = set(grn.edges["target"])
    master = next((gene for gene in grn.genes if gene not in incoming), _safe_gene(grn, "g0", 0))
    activation_rows = grn.edges.loc[grn.edges["sign"] == "activation", "target"]
    repression_rows = grn.edges.loc[grn.edges["sign"] == "repression", "target"]
    activation = str(activation_rows.iloc[0]) if not activation_rows.empty else _safe_gene(grn, "g4", 4)
    repression = str(repression_rows.iloc[0]) if not repression_rows.empty else _safe_gene(grn, "g5", 5)
    return {
        "master": master,
        "activation_target": activation,
        "repression_target": repression,
    }


def _simulate_example(grn, capture_rate: float, dropout_rate: float, poisson_observed: bool = True):
    return simulate_linear(
        grn,
        n_cells=100,
        time_end=4.0,
        dt=0.02,
        master_programs={
            "g0": linear_increase(0.2, 1.2),
            "g1": 0.8,
            "g2": sigmoid_decrease(1.1, 0.2),
        },
        seed=42,
        capture_rate=capture_rate,
        poisson_observed=poisson_observed,
        dropout_rate=dropout_rate,
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
    base_dir = Path(__file__).with_name("outputs") / "linear_20gene" / "plots"
    if base_dir.exists():
        shutil.rmtree(base_dir)
    true_dir = base_dir / "true"
    observed_dir = base_dir / "observed"
    lownoise_dir = base_dir / "observed_lownoise"
    diagnostics_dir = base_dir / "diagnostics"
    for directory in (true_dir, observed_dir, lownoise_dir, diagnostics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    grn = build_example_grn()
    noisy = _simulate_example(grn, capture_rate=0.5, dropout_rate=0.02)
    lownoise = _simulate_example(grn, capture_rate=1.0, dropout_rate=0.0, poisson_observed=False)
    selected = select_representative_genes(grn)

    generated: list[Path] = []

    generated.append(true_dir / "embedding_pca_true_by_pseudotime.png")
    plot_embedding_by_pseudotime(noisy, method="pca", layer_preference="true", output_path=generated[-1])
    generated.append(true_dir / "embedding_pca_true_with_velocity.png")
    plot_embedding_with_velocity(noisy, method="pca", layer_preference="true", output_path=generated[-1])
    for label, gene in selected.items():
        generated.append(true_dir / f"phase_portrait_{label}_{gene}_true.png")
        plot_phase_portrait(noisy, gene, mode="true", output_path=generated[-1])
        generated.append(true_dir / f"gene_dynamics_{label}_{gene}.png")
        plot_gene_dynamics_over_pseudotime(noisy, gene, output_path=generated[-1])

    generated.append(observed_dir / "embedding_pca_observed_by_pseudotime.png")
    plot_embedding_by_pseudotime(noisy, method="pca", layer_preference="observed", output_path=generated[-1])
    _, umap_method, _ = compute_umap_embedding(noisy, random_state=42, layer_preference="observed")
    umap_available = umap_method == "umap"
    if umap_available:
        generated.append(observed_dir / "embedding_umap_observed_by_pseudotime.png")
        plot_embedding_by_pseudotime(
            noisy,
            method="umap",
            layer_preference="observed",
            random_state=42,
            output_path=generated[-1],
        )
    else:
        print("umap-learn is unavailable; skipping observed UMAP plot")
    for label, gene in selected.items():
        path = _save_observed_phase(noisy, gene, label, observed_dir, "observed")
        if path is not None:
            generated.append(path)

    generated.append(lownoise_dir / "embedding_pca_observed_lownoise_by_pseudotime.png")
    plot_embedding_by_pseudotime(lownoise, method="pca", layer_preference="observed", output_path=generated[-1])
    for label, gene in selected.items():
        path = _save_observed_phase(lownoise, gene, label, lownoise_dir, "observed_lownoise")
        if path is not None:
            generated.append(path)

    (diagnostics_dir / "README.txt").write_text(
        "Primary checks live in true/. Observed plots inspect technical noise. "
        "observed_lownoise/ uses capture_rate=1.0, dropout_rate=0.0, and poisson_observed=False "
        "for visualization/debugging rather than realistic UMI noise.\n",
        encoding="utf-8",
    )

    plt.close("all")
    print(f"umap_available={umap_available}")
    print(f"saved plots to {base_dir}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
