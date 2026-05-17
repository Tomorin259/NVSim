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
from matplotlib.lines import Line2D
from sklearn.neighbors import NearestNeighbors

from .output import to_anndata


def _is_anndata(obj: Any) -> bool:
    return hasattr(obj, "layers") and hasattr(obj, "obs") and hasattr(obj, "var")


def _as_anndata(data: Any, *, copy: bool = True):
    if _is_anndata(data):
        return data.copy() if copy else data
    if isinstance(data, dict):
        return to_anndata(data)
    raise TypeError("data must be an NVSim result dict or AnnData")


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

    group_key = "state" if "state" in adata.obs else ("branch" if "branch" in adata.obs else None)
    if group_key is not None:
        labels = adata.obs[group_key].astype(str).to_numpy()
        groups = pd.unique(labels)
        if len(groups) >= 2:
            alpha_means = np.vstack([alpha[labels == group].mean(axis=0) for group in groups])
            spliced_means = np.vstack([spliced[labels == group].mean(axis=0) for group in groups])
            alpha_gap = alpha_means.max(axis=0) - alpha_means.min(axis=0)
            spliced_gap = spliced_means.max(axis=0) - spliced_means.min(axis=0)
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
        color="state" if "state" in adata.obs else ("branch" if "branch" in adata.obs else None),
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
    dpi: int = 180,
):
    """Plot selected gene quantities over pseudotime, grouped by branch."""

    adata = _as_anndata(data, copy=True)
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
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_gene_dynamics_over_pseudotime(data: Any, gene: str | int, output_path: str | Path | None = None, **kwargs):
    """Backward-compatible wrapper around ``plot_gene_dynamics``."""

    return plot_gene_dynamics(data, [gene], output_path=output_path, **kwargs)



def _sample_phase_portrait_indices(
    points: np.ndarray,
    *,
    step: tuple[int, int] = (30, 30),
    percentile: float = 15.0,
    jitter_scale: float = 0.15,
    kernel_sigma: float = 0.5,
    random_state: int = 10,
) -> np.ndarray:
    """Select representative phase-portrait anchor points on a coarse grid.

    This is a repository-local port of the public example helper previously
    imported from an external private path. The sampler perturbs a regular grid,
    snaps grid points to nearby observations, and filters anchors in low-density
    regions.
    """

    coords = np.asarray(points, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("points must be a 2D array with exactly two columns")
    n_obs = coords.shape[0]
    if n_obs == 0:
        return np.array([], dtype=int)
    if n_obs == 1:
        return np.array([0], dtype=int)

    grids: list[np.ndarray] = []
    for dim, n_steps in enumerate(step):
        lower = float(np.min(coords[:, dim]))
        upper = float(np.max(coords[:, dim]))
        pad = 0.025 * abs(upper - lower)
        if pad == 0.0:
            pad = 0.025
        grids.append(np.linspace(lower - pad, upper + pad, int(n_steps)))

    mesh = np.meshgrid(*grids)
    grid_points = np.vstack([axis.ravel() for axis in mesh]).T
    rng = np.random.default_rng(random_state)
    grid_points = grid_points + rng.normal(loc=0.0, scale=jitter_scale, size=grid_points.shape)

    first_k = min(max(n_obs - 1, 1), 20)
    nn = NearestNeighbors()
    nn.fit(coords)
    _, neighbor_ix = nn.kneighbors(grid_points, first_k)
    chosen = np.unique(neighbor_ix[:, 0].ravel())
    if chosen.size == 0:
        return np.array([], dtype=int)

    second_k = min(max(n_obs - 1, 1), 20)
    nn = NearestNeighbors()
    nn.fit(coords)
    distances, _ = nn.kneighbors(coords[chosen], second_k)

    density = np.exp(-(distances**2) / (2.0 * kernel_sigma**2)) / np.sqrt(2.0 * np.pi * kernel_sigma**2)
    density = density.sum(axis=1)
    keep = density > np.percentile(density, percentile)
    return chosen[keep]

def plot_phase_portrait_gallery(*args, **kwargs):
    """Backward-compatible wrapper around ``plot_phase_gallery``."""

    return plot_phase_gallery(*args, **kwargs)


def _plot_gene_dynamics(adata: Any, genes: Iterable[str], output_path: Path, dpi: int) -> None:
    fig = plot_gene_dynamics(adata, list(genes), output_path=None, dpi=dpi)
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
    *,
    layer_s: str | None = None,
    layer_u: str | None = None,
    layer_v_s: str | None = None,
    layer_v_u: str | None = None,
    color_key: str | None = None,
    s: float = 18,
    alpha: float = 0.82,
    arrow_grid: tuple[int, int] = (30, 30),
    legend_loc: str | None = "upper right",
    velocity_percentile: float = 15.0,
    velocity_color: str = "k",
    velocity_width: float = 0.002,
    velocity_headwidth: float = 4.5,
    velocity_headlength: float = 5.0,
    velocity_headaxislength: float = 4.5,
    show_anchor_points: bool = True,
):
    """Plot a gene phase portrait with optional 2D RNA velocity arrows.

    The public API supports both the original NVSim ``mode=`` interface and the
    more explicit layer-based interface used by DS6 example workflows.
    """

    adata = _as_anndata(data, copy=True)
    names = [str(name) for name in adata.var_names]
    idx = int(gene) if isinstance(gene, int) else names.index(str(gene))
    gene_name = names[idx]

    if layer_s is None or layer_u is None:
        s_mat, u_mat, ds_mat, du_mat = _phase_layers(adata, mode=mode)
        if layer_s is None:
            layer_s = "true_spliced" if mode == "true" else "spliced"
        if layer_u is None:
            layer_u = "true_unspliced" if mode == "true" else "unspliced"
        if layer_v_s is None:
            layer_v_s = "true_velocity" if "true_velocity" in adata.layers else "velocity"
        if layer_v_u is None and du_mat is not None:
            layer_v_u = "true_velocity_u"
    else:
        s_mat = _require_layer(adata, layer_s)
        u_mat = _require_layer(adata, layer_u)
        ds_mat = _require_layer(adata, layer_v_s) if layer_v_s and layer_v_s in adata.layers else None
        du_mat = _require_layer(adata, layer_v_u) if layer_v_u and layer_v_u in adata.layers else None

    S = np.asarray(s_mat[:, idx], dtype=float)
    U = np.asarray(u_mat[:, idx], dtype=float)
    V_S = np.asarray(ds_mat[:, idx], dtype=float) if ds_mat is not None else np.zeros_like(S)
    V_U = np.asarray(du_mat[:, idx], dtype=float) if du_mat is not None else np.zeros_like(U)

    color_field = color_key if color_key is not None else color_by

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(4.6, 4.0))
    else:
        fig = ax.figure

    if color_field and color_field in adata.obs:
        series = adata.obs[color_field]
        if pd.api.types.is_numeric_dtype(series):
            scatter = ax.scatter(S, U, c=series.to_numpy(dtype=float), cmap="viridis", s=s, alpha=alpha, edgecolors="none")
            if created_fig:
                fig.colorbar(scatter, ax=ax, label=color_field)
        else:
            labels = series.astype(str)
            if hasattr(series, "cat"):
                ordered = [str(v) for v in series.cat.categories if str(v) in set(labels)]
            else:
                ordered = [str(v) for v in pd.unique(labels)]
            palette = adata.uns.get(f"{color_field}_colors")
            if palette is None and color_field == "state":
                palette = adata.uns.get("state_colors")
            if palette is None and color_field == "clusters":
                palette = adata.uns.get("clusters_colors")
            if palette is None:
                cmap = plt.get_cmap("tab20", max(len(ordered), 1))
                color_map = {cat: cmap(i) for i, cat in enumerate(ordered)}
            else:
                color_map = {cat: palette[i] if i < len(palette) else "grey" for i, cat in enumerate(ordered)}
            colors = [color_map[str(label)] for label in labels]
            ax.scatter(S, U, c=colors, s=s, alpha=alpha, edgecolors="none")
            if legend_loc:
                handles = [Line2D([0], [0], marker="o", color="w", label=cat, markerfacecolor=color_map[cat], markersize=8) for cat in ordered]
                ax.legend(handles=handles, loc=legend_loc, bbox_to_anchor=(1.05, 1), frameon=False)
    else:
        ax.scatter(S, U, c="#95D9EF", s=s, alpha=alpha, edgecolors="none")

    if show_velocity:
        if arrow_stride is not None:
            order = np.argsort(adata.obs["pseudotime"].to_numpy(dtype=float)) if "pseudotime" in adata.obs else np.arange(adata.n_obs)
            arrow_ix = order[:: max(1, int(arrow_stride))]
        else:
            arrow_ix = _sample_phase_portrait_indices(np.column_stack([U, S]), step=arrow_grid, percentile=velocity_percentile)
        if arrow_ix.size > 0:
            if show_anchor_points:
                ax.scatter(S[arrow_ix], U[arrow_ix], color="none", edgecolor=velocity_color, s=s, linewidth=0.8, zorder=19)
            ax.quiver(
                S[arrow_ix],
                U[arrow_ix],
                V_S[arrow_ix],
                V_U[arrow_ix],
                angles="xy",
                scale_units="xy",
                scale=None,
                color=velocity_color,
                width=velocity_width,
                headwidth=velocity_headwidth,
                headlength=velocity_headlength,
                headaxislength=velocity_headaxislength,
                alpha=1.0,
                zorder=20,
            )

    ax.set_xlabel(f"Spliced ({layer_s})")
    ax.set_ylabel(f"Unspliced ({layer_u})")
    ax.set_title(f"Gene: {gene_name}")

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
    return fig if created_fig else ax


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

    adata = _as_anndata(data, copy=True)
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
