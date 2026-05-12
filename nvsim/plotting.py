"""scanpy/scVelo-based plotting and mechanistic diagnostics for NVSim.

This is the single public plotting module. scanpy handles PCA/neighbors/UMAP,
scVelo handles velocity graph and velocity stream visualization, and NVSim keeps
small mechanistic diagnostics such as phase portraits and gene dynamics.
"""

from __future__ import annotations

from pathlib import Path
import json
import warnings
from typing import Any, Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from .output import to_anndata


def _is_anndata(obj: Any) -> bool:
    return hasattr(obj, "layers") and hasattr(obj, "obs") and hasattr(obj, "var")


def _dense_array(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D cells x genes matrix")
    return arr.copy()


def _require_layer(adata: Any, layer: str) -> np.ndarray:
    if layer not in adata.layers:
        raise KeyError(f"missing required AnnData layer {layer!r}")
    return _dense_array(adata.layers[layer], name=f"adata.layers[{layer!r}]")


def prepare_adata(
    data: Any,
    expression_layer: str = "true",
    velocity_layer: str = "true_velocity",
    copy: bool = True,
):
    """Prepare AnnData for scanpy/scVelo velocity visualization.

    Parameters
    ----------
    data:
        NVSim result dict or AnnData. Result dicts are first converted with
        ``nvsim.output.to_anndata``.
    expression_layer:
        ``"true"`` uses simulator truth layers: ``true_spliced`` for ``X`` and
        ``spliced``, and ``true_unspliced`` for ``unspliced``.
        ``"observed"`` uses noisy observed ``spliced``/``unspliced`` layers.
    velocity_layer:
        Source layer copied into ``adata.layers["velocity"]``. The default
        ``"true_velocity"`` is the simulator ground-truth ``ds/dt`` RNA
        velocity, not an inferred velocity.
    copy:
        Copy AnnData inputs before modifying them.
    """

    if expression_layer not in {"true", "observed"}:
        raise ValueError("expression_layer must be 'true' or 'observed'")
    if _is_anndata(data):
        adata = data.copy() if copy else data
    elif isinstance(data, dict):
        adata = to_anndata(data)
        if not copy:
            # Result dict conversion always creates a new AnnData object.
            pass
    else:
        raise TypeError("data must be an NVSim result dict or AnnData")

    if expression_layer == "true":
        spliced_key = "true_spliced"
        unspliced_key = "true_unspliced"
    else:
        spliced_key = "spliced"
        unspliced_key = "unspliced"

    spliced = _require_layer(adata, spliced_key)
    unspliced = _require_layer(adata, unspliced_key)
    velocity = _require_layer(adata, velocity_layer)

    expected = spliced.shape
    for key, arr in [(unspliced_key, unspliced), (velocity_layer, velocity)]:
        if arr.shape != expected:
            raise ValueError(f"layer {key!r} has shape {arr.shape}, expected {expected}")

    adata.X = spliced.copy()
    adata.layers["spliced"] = spliced.copy()
    adata.layers["unspliced"] = unspliced.copy()
    adata.layers["velocity"] = velocity.copy()
    adata.uns.setdefault("nvsim_velocity_showcase", {})
    adata.uns["nvsim_velocity_showcase"].update(
        {
            "expression_layer": expression_layer,
            "selected_spliced_layer": spliced_key,
            "selected_unspliced_layer": unspliced_key,
            "velocity_layer": velocity_layer,
            "velocity_kind": "ground_truth" if velocity_layer == "true_velocity" else "custom",
        }
    )
    return adata


def embed(
    adata: Any,
    n_pcs: int = 20,
    n_neighbors: int = 15,
    min_dist: float = 0.3,
    basis: str = "umap",
    scale: bool = False,
    log1p: bool = False,
    random_state: int = 0,
    copy: bool = False,
):
    """Run scanpy PCA, neighbors, and UMAP on the selected expression matrix.

    NVSim true layers are already simulator states rather than ordinary count
    matrices, so this function does not normalize total counts by default.
    """

    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError("scanpy is required for embed") from exc

    if basis != "umap":
        raise ValueError("currently only basis='umap' is supported for scanpy embedding")
    out = adata.copy() if copy else adata
    if out.n_obs < 3:
        raise ValueError("at least three cells are required for scanpy neighbors/UMAP")
    if out.n_vars < 2:
        raise ValueError("at least two genes are required for scanpy PCA")

    n_pcs_use = min(int(n_pcs), out.n_obs - 1, out.n_vars - 1)
    if n_pcs_use < 1:
        raise ValueError("n_pcs is too small after shape adjustment")

    if log1p:
        sc.pp.log1p(out)
    if scale:
        sc.pp.scale(out)
    sc.pp.pca(out, n_comps=n_pcs_use, random_state=random_state, svd_solver="arpack")
    neighbors_use = min(int(n_neighbors), out.n_obs - 1)
    if neighbors_use < 2:
        raise ValueError("n_neighbors is too small after shape adjustment")
    sc.pp.neighbors(out, n_neighbors=neighbors_use, n_pcs=n_pcs_use, random_state=random_state)
    sc.tl.umap(out, min_dist=min_dist, random_state=random_state)
    return out


def velocity_stream(
    adata: Any,
    basis: str = "umap",
    velocity_layer: str = "velocity",
    recompute_graph: bool = True,
):
    """Build scVelo velocity graph/embedding from an existing velocity layer.

    ``adata.layers[velocity_layer]`` is assumed to already contain NVSim
    ground-truth velocity, usually ``true_velocity`` copied to ``velocity`` by
    ``prepare_adata``. This function does not estimate velocity with
    scVelo stochastic or dynamical models.
    """

    try:
        import scvelo as scv
    except ImportError as exc:
        raise ImportError("scvelo is required for velocity_stream") from exc

    if velocity_layer not in adata.layers:
        raise KeyError(f"missing velocity layer {velocity_layer!r}")
    if f"X_{basis}" not in adata.obsm:
        raise KeyError(f"missing embedding adata.obsm['X_{basis}']; run embed first")
    if velocity_layer != "velocity":
        adata.layers["velocity"] = _dense_array(adata.layers[velocity_layer], name=velocity_layer)
        vkey = "velocity"
    else:
        adata.layers["velocity"] = _dense_array(adata.layers["velocity"], name="velocity")
        vkey = "velocity"

    try:
        if recompute_graph or f"{vkey}_graph" not in adata.uns:
            scv.tl.velocity_graph(adata, vkey=vkey)
        scv.tl.velocity_embedding(adata, basis=basis, vkey=vkey)
    except Exception as exc:  # scVelo raises several implementation-specific errors.
        raise RuntimeError(
            "scVelo failed to build a velocity graph/embedding from the supplied "
            f"ground-truth velocity layer {velocity_layer!r}: {exc}"
        ) from exc
    return adata


def _grn_dataframe(adata: Any) -> pd.DataFrame:
    raw = adata.uns.get("true_grn")
    if raw is None:
        return pd.DataFrame()
    if isinstance(raw, pd.DataFrame):
        return raw.copy()
    if isinstance(raw, dict) and {"columns", "data"}.issubset(raw):
        index = raw.get("index")
        return pd.DataFrame(raw["data"], columns=raw["columns"], index=index)
    try:
        return pd.DataFrame(raw)
    except Exception:
        return pd.DataFrame()


def select_genes(adata: Any, representative_genes: Any = "auto") -> list[str]:
    """Select stable representative genes for velocity showcase panels."""

    genes = [str(gene) for gene in adata.var_names]
    if representative_genes != "auto":
        selected = [str(gene) for gene in representative_genes]
        missing = [gene for gene in selected if gene not in genes]
        if missing:
            raise KeyError(f"representative genes not found: {missing}")
        return selected

    selected: list[str] = []

    def add(gene: str | None) -> None:
        if gene is not None and gene in genes and gene not in selected:
            selected.append(gene)

    alpha = np.asarray(adata.layers.get("true_alpha", adata.layers["velocity"]), dtype=float)
    spliced = np.asarray(adata.layers.get("true_spliced", adata.layers["spliced"]), dtype=float)
    velocity = np.asarray(adata.layers["velocity"], dtype=float)
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}

    if "gene_role" in adata.var.columns:
        masters = [str(g) for g in adata.var_names[adata.var["gene_role"].astype(str) == "master_regulator"]]
        if masters:
            add(max(masters, key=lambda gene: float(np.var(alpha[:, gene_to_idx[gene]]))))

    branch = adata.obs["branch"].astype(str).to_numpy() if "branch" in adata.obs else np.array(["all"] * adata.n_obs)
    b0 = branch == "branch_0"
    b1 = branch == "branch_1"
    if b0.any() and b1.any():
        alpha_gap = np.abs(alpha[b0].mean(axis=0) - alpha[b1].mean(axis=0))
        spliced_gap = np.abs(spliced[b0].mean(axis=0) - spliced[b1].mean(axis=0))
        add(genes[int(np.argmax(alpha_gap + spliced_gap))])

    add(genes[int(np.argmax(np.var(velocity, axis=0)))])

    grn = _grn_dataframe(adata)
    if not grn.empty and {"target", "sign"}.issubset(grn.columns):
        for sign in ("activation", "repression"):
            targets = [str(gene) for gene in grn.loc[grn["sign"].astype(str) == sign, "target"] if str(gene) in gene_to_idx]
            if targets:
                add(max(targets, key=lambda gene: float(np.var(alpha[:, gene_to_idx[gene]]))))

    if len(selected) < 3:
        warnings.warn("falling back to leading genes for representative gene selection", UserWarning, stacklevel=2)
        for gene in genes:
            add(gene)
            if len(selected) >= 4:
                break
    return selected[:4]


def _embedding(adata: Any, basis: str) -> np.ndarray:
    key = f"X_{basis}"
    if key not in adata.obsm:
        raise KeyError(f"missing embedding adata.obsm[{key!r}]")
    coords = np.asarray(adata.obsm[key], dtype=float)
    if coords.shape[1] < 2:
        raise ValueError(f"embedding {key!r} must have at least two dimensions")
    return coords[:, :2]


def _plot_embedding_by_category(adata: Any, output_path: Path, basis: str, color: str, title: str, dpi: int) -> None:
    coords = _embedding(adata, basis)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    if color not in adata.obs:
        ax.scatter(coords[:, 0], coords[:, 1], s=18, alpha=0.8)
    elif pd.api.types.is_numeric_dtype(adata.obs[color]):
        values = adata.obs[color].to_numpy(dtype=float)
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, s=18, alpha=0.85, cmap="viridis")
        fig.colorbar(sc, ax=ax, label=color)
    else:
        categories = adata.obs[color].astype(str).to_numpy()
        for idx, category in enumerate(pd.unique(categories)):
            mask = categories == category
            ax.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.85, label=category)
        ax.legend(frameon=False, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(f"{basis.upper()}1")
    ax.set_ylabel(f"{basis.upper()}2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_velocity_stream(adata: Any, output_path: Path, basis: str, dpi: int) -> None:
    try:
        import scvelo as scv
    except ImportError as exc:
        raise ImportError("scvelo is required for velocity stream plotting") from exc

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    scv.pl.velocity_embedding_stream(
        adata,
        basis=basis,
        vkey="velocity",
        color="branch" if "branch" in adata.obs else None,
        title="Ground-truth RNA velocity stream",
        show=False,
        ax=ax,
        legend_loc="right margin",
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_gene_dynamics(
    data: Any,
    genes: Iterable[str] | str,
    output_path: str | Path | None = None,
    quantities: tuple[str, ...] = ("true_alpha", "true_unspliced", "true_spliced", "true_velocity", "true_velocity_u"),
):
    """Plot selected gene quantities over pseudotime, grouped by branch."""

    adata = prepare_adata(data, expression_layer="true") if not _is_anndata(data) else data.copy()
    gene_list = [genes] if isinstance(genes, (str, int)) else list(genes)
    names = [str(name) for name in adata.var_names]
    pt = adata.obs["pseudotime"].to_numpy(dtype=float) if "pseudotime" in adata.obs else np.arange(adata.n_obs)
    branch = adata.obs["branch"].astype(str).to_numpy() if "branch" in adata.obs else np.array(["all"] * adata.n_obs)
    fig, axes = plt.subplots(len(gene_list), len(quantities), figsize=(3.4 * len(quantities), 2.3 * len(gene_list)), squeeze=False)
    for row, gene in enumerate(gene_list):
        idx = int(gene) if isinstance(gene, int) else names.index(str(gene))
        for col, layer in enumerate(quantities):
            ax = axes[row, col]
            if layer not in adata.layers:
                ax.text(0.5, 0.5, f"missing {layer}", ha="center", va="center")
                ax.set_axis_off()
                continue
            values = np.asarray(adata.layers[layer], dtype=float)[:, idx]
            for category in pd.unique(branch):
                mask = branch == category
                order = np.argsort(pt[mask])
                ax.plot(pt[mask][order], values[mask][order], ".", markersize=3, alpha=0.75, label=category)
            if row == 0:
                ax.set_title(layer)
            if col == 0:
                ax.set_ylabel(str(gene))
            if row == len(gene_list) - 1:
                ax.set_xlabel("pseudotime")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, -1].legend(handles, labels, frameon=False, fontsize=8)
    fig.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
    return fig


def plot_gene_dynamics_over_pseudotime(data: Any, gene: str | int, output_path: str | Path | None = None, **kwargs):
    """Backward-compatible wrapper around ``plot_gene_dynamics``."""

    return plot_gene_dynamics(data, [gene], output_path=output_path, **kwargs)


def plot_phase_portrait_gallery(*args, **kwargs):
    """Backward-compatible wrapper around ``plot_phase_gallery``."""

    return plot_phase_gallery(*args, **kwargs)


def _plot_gene_dynamics(adata: Any, genes: Iterable[str], output_path: Path, dpi: int) -> None:
    genes = list(genes)
    layers = [("true_alpha", "alpha"), ("true_spliced", "spliced"), ("velocity", "velocity")]
    pt = adata.obs["pseudotime"].to_numpy(dtype=float) if "pseudotime" in adata.obs else np.arange(adata.n_obs)
    branch = adata.obs["branch"].astype(str).to_numpy() if "branch" in adata.obs else np.array(["all"] * adata.n_obs)
    fig, axes = plt.subplots(len(genes), len(layers), figsize=(4.0 * len(layers), 2.3 * len(genes)), squeeze=False)
    for row, gene in enumerate(genes):
        idx = list(map(str, adata.var_names)).index(str(gene))
        for col, (layer, label) in enumerate(layers):
            ax = axes[row, col]
            if layer not in adata.layers:
                ax.text(0.5, 0.5, f"missing {layer}", ha="center", va="center")
                ax.set_axis_off()
                continue
            values = np.asarray(adata.layers[layer], dtype=float)[:, idx]
            for category in pd.unique(branch):
                mask = branch == category
                order = np.argsort(pt[mask])
                ax.plot(pt[mask][order], values[mask][order], ".", markersize=3, alpha=0.75, label=category)
            if row == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(f"{gene}")
            if row == len(genes) - 1:
                ax.set_xlabel("pseudotime")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, -1].legend(handles, labels, frameon=False, fontsize=8)
    fig.suptitle("Representative gene dynamics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _phase_layers(adata: Any, mode: str = "true") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    if mode not in {"true", "observed"}:
        raise ValueError("mode must be 'true' or 'observed'")
    if mode == "true":
        s = _require_layer(adata, "true_spliced")
        u = _require_layer(adata, "true_unspliced")
    else:
        s = _require_layer(adata, "spliced")
        u = _require_layer(adata, "unspliced")
    ds = _require_layer(adata, "true_velocity") if "true_velocity" in adata.layers else _require_layer(adata, "velocity")
    du = _require_layer(adata, "true_velocity_u") if "true_velocity_u" in adata.layers else None
    return s, u, ds, du


def plot_phase_portrait(
    data: Any,
    gene: str | int,
    output_path: str | Path | None = None,
    ax=None,
    mode: str = "true",
    color_by: str = "pseudotime",
    show_velocity: bool = True,
    arrow_stride: int | None = None,
):
    """Plot a gene phase portrait with true 2D RNA velocity arrows.

    The x-axis is spliced RNA and the y-axis is unspliced RNA. For true layers,
    arrows use ``dx=true_velocity=ds/dt`` and ``dy=true_velocity_u=du/dt``.
    """

    adata = prepare_adata(data, expression_layer="true") if not _is_anndata(data) else data.copy()
    names = [str(name) for name in adata.var_names]
    idx = int(gene) if isinstance(gene, int) else names.index(str(gene))
    s, u, ds, du = _phase_layers(adata, mode=mode)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
    else:
        fig = ax.figure
    if color_by in adata.obs:
        values = adata.obs[color_by].to_numpy()
        if pd.api.types.is_numeric_dtype(adata.obs[color_by]):
            pts = ax.scatter(s[:, idx], u[:, idx], c=values.astype(float), cmap="viridis", s=18, alpha=0.82)
            fig.colorbar(pts, ax=ax, label=color_by)
        else:
            cats = adata.obs[color_by].astype(str).to_numpy()
            for category in pd.unique(cats):
                mask = cats == category
                ax.scatter(s[mask, idx], u[mask, idx], s=18, alpha=0.82, label=category)
            ax.legend(frameon=False, fontsize=8)
    else:
        ax.scatter(s[:, idx], u[:, idx], s=18, alpha=0.82)
    if show_velocity and du is not None:
        order = np.argsort(adata.obs["pseudotime"].to_numpy(dtype=float)) if "pseudotime" in adata.obs else np.arange(adata.n_obs)
        stride = arrow_stride or max(1, len(order) // 30)
        q_idx = order[::stride]
        ax.quiver(
            s[q_idx, idx],
            u[q_idx, idx],
            ds[q_idx, idx],
            du[q_idx, idx],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.003,
            color="0.25",
            alpha=0.6,
        )
    ax.set_title(f"Phase portrait: {names[idx]}")
    ax.set_xlabel("spliced RNA")
    ax.set_ylabel("unspliced RNA")
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
    return fig


def plot_phase_gallery(
    data: Any,
    genes: Iterable[str] | None = None,
    output_path: str | Path | None = None,
    mode: str = "true",
    color_by: str | None = None,
    max_cols: int = 5,
    panel_size: float = 2.2,
):
    """Plot a thumbnail grid of gene phase portraits."""

    adata = prepare_adata(data, expression_layer="true") if not _is_anndata(data) else data.copy()
    gene_names = [str(g) for g in adata.var_names] if genes is None else [str(g) for g in genes]
    if not gene_names:
        raise ValueError("at least one gene is required")
    n_cols = min(max_cols, len(gene_names))
    n_rows = int(np.ceil(len(gene_names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_size * n_cols, panel_size * n_rows), squeeze=False)
    for ax, gene in zip(axes.ravel(), gene_names):
        plot_phase_portrait(adata, gene, ax=ax, mode=mode, color_by=color_by or "", show_velocity=False)
        ax.set_title(str(gene), fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(gene_names):]:
        ax.axis("off")
    fig.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
    return fig


def _plot_phase_portraits(adata: Any, genes: Iterable[str], output_path: Path, dpi: int) -> None:
    genes = list(genes)
    fig = plot_phase_gallery(adata, genes=genes, mode="true", max_cols=len(genes), output_path=None)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_showcase_panel(adata: Any, output_path: Path, basis: str, dpi: int) -> None:
    coords = _embedding(adata, basis)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    branch = adata.obs["branch"].astype(str).to_numpy() if "branch" in adata.obs else np.array(["all"] * adata.n_obs)
    for category in pd.unique(branch):
        mask = branch == category
        axes[0].scatter(coords[mask, 0], coords[mask, 1], s=14, alpha=0.85, label=category)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Branch")
    pt = adata.obs["pseudotime"].to_numpy(dtype=float) if "pseudotime" in adata.obs else np.arange(adata.n_obs)
    sc = axes[1].scatter(coords[:, 0], coords[:, 1], c=pt, s=14, alpha=0.85, cmap="viridis")
    fig.colorbar(sc, ax=axes[1], label="pseudotime")
    axes[1].set_title("Pseudotime")
    if f"velocity_{basis}" in adata.obsm:
        vel = np.asarray(adata.obsm[f"velocity_{basis}"], dtype=float)
        stride = max(1, adata.n_obs // 80)
        axes[2].scatter(coords[:, 0], coords[:, 1], c="0.82", s=8)
        axes[2].quiver(
            coords[::stride, 0],
            coords[::stride, 1],
            vel[::stride, 0],
            vel[::stride, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.003,
            color="0.2",
        )
        axes[2].set_title("Ground-truth velocity")
    else:
        axes[2].text(0.5, 0.5, "velocity embedding unavailable", ha="center", va="center")
        axes[2].set_axis_off()
    for ax in axes[:2]:
        ax.set_xlabel(f"{basis.upper()}1")
        ax.set_ylabel(f"{basis.upper()}2")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_showcase(
    data: Any,
    output_dir: str | Path,
    expression_layer: str = "true",
    basis: str = "umap",
    n_pcs: int = 20,
    n_neighbors: int = 15,
    min_dist: float = 0.3,
    representative_genes: Any = "auto",
    dpi: int = 300,
    random_state: int = 0,
) -> dict[str, Any]:
    """Create a scanpy/scVelo RNA velocity-style showcase panel.

    The stream plot is a ground-truth velocity stream based on NVSim
    ``true_velocity`` copied to ``adata.layers["velocity"]``. It is not scVelo
    inferred velocity.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "expression_layer": expression_layer,
        "velocity_layer": "true_velocity",
        "basis": basis,
        "scanpy_embedding": "not_run",
        "scvelo_velocity": "not_run",
        "files": {},
    }

    adata = prepare_adata(data, expression_layer=expression_layer, velocity_layer="true_velocity", copy=True)
    embed(
        adata,
        n_pcs=n_pcs,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        basis=basis,
        random_state=random_state,
        copy=False,
    )
    summary["scanpy_embedding"] = "ok"

    branch_path = output_path / "embedding_by_branch.png"
    _plot_embedding_by_category(adata, branch_path, basis, "branch", "Expression manifold by branch", dpi)
    summary["files"]["embedding_by_branch"] = str(branch_path)

    pt_path = output_path / "embedding_by_pseudotime.png"
    _plot_embedding_by_category(adata, pt_path, basis, "pseudotime", "Expression manifold by pseudotime", dpi)
    summary["files"]["embedding_by_pseudotime"] = str(pt_path)

    try:
        velocity_stream(adata, basis=basis, velocity_layer="velocity")
        stream_path = output_path / "velocity_stream_true.png"
        _plot_velocity_stream(adata, stream_path, basis, dpi)
        summary["scvelo_velocity"] = "ok"
        summary["files"]["velocity_stream_true"] = str(stream_path)
    except Exception as exc:
        msg = str(exc)
        warnings.warn(msg, UserWarning, stacklevel=2)
        (output_path / "velocity_stream_true_ERROR.txt").write_text(msg + "\n", encoding="utf-8")
        summary["scvelo_velocity"] = "failed"
        summary["scvelo_error"] = msg

    selected = select_genes(adata, representative_genes=representative_genes)
    summary["representative_genes"] = selected

    dynamics_path = output_path / "gene_dynamics_representative.png"
    _plot_gene_dynamics(adata, selected, dynamics_path, dpi)
    summary["files"]["gene_dynamics_representative"] = str(dynamics_path)

    phase_path = output_path / "phase_portrait_representative.png"
    _plot_phase_portraits(adata, selected, phase_path, dpi)
    summary["files"]["phase_portrait_representative"] = str(phase_path)

    panel_path = output_path / "velocity_showcase_panel.png"
    _plot_showcase_panel(adata, panel_path, basis, dpi)
    summary["files"]["velocity_showcase_panel"] = str(panel_path)

    summary_path = output_path / "showcase_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["files"]["showcase_summary"] = str(summary_path)
    return summary


# Descriptive aliases retained for readability and backward compatibility.
prepare_velocity_adata = prepare_adata
run_scanpy_embedding = embed
run_scvelo_velocity_stream = velocity_stream
select_velocity_showcase_genes = select_genes
plot_velocity_showcase = plot_showcase
