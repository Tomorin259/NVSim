"""NVSim 结果快速检查用的轻量 matplotlib 绘图工具。

约定：
- 所有 layer 矩阵都是 cells x genes。
- phase portrait 默认使用 true layers，因为 observed UMI 在 capture/Poisson
  后可能非常离散，不适合检查真实动力学。
- observed UMAP 在稀疏/噪声条件下可能把连续轨迹切碎；科学检查应优先看
  true PCA、true velocity arrows 和 gene dynamics。
- velocity arrows 是 PCA 投影的 quick-look 诊断，不是完整 scVelo-style
  velocity embedding。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


VALID_LAYER_PREFERENCES = {"observed", "true", "auto"}
DEFAULT_BRANCH_COLORS = {
    "trunk": "tab:blue",
    "branch_0": "tab:orange",
    "branch_1": "tab:green",
    "root": "tab:blue",
    "all": "tab:blue",
}


def _is_anndata(obj: Any) -> bool:
    return hasattr(obj, "layers") and hasattr(obj, "obs") and hasattr(obj, "var")


def _layers(data: Any) -> dict[str, np.ndarray]:
    """统一读取 dict 或 AnnData 中的 layers。"""

    if _is_anndata(data):
        return {key: np.asarray(value) for key, value in data.layers.items()}
    return data["layers"]


def _obs(data: Any) -> pd.DataFrame:
    return data.obs.copy() if _is_anndata(data) else data["obs"].copy()


def _var_names(data: Any) -> list[str]:
    if _is_anndata(data):
        return [str(name) for name in data.var_names]
    var = data["var"]
    return [str(name) for name in var.index]


def _matrix(data: Any, layer: str) -> np.ndarray:
    layers = _layers(data)
    if layer not in layers:
        raise KeyError(f"missing layer {layer!r}")
    return np.asarray(layers[layer], dtype=float)


def _spliced_layer_name(data: Any, layer_preference: str) -> str:
    """根据 layer_preference 明确选择 true 或 observed spliced layer。"""

    if layer_preference not in VALID_LAYER_PREFERENCES:
        raise ValueError(f"layer_preference must be one of {sorted(VALID_LAYER_PREFERENCES)}")
    layers = _layers(data)
    if layer_preference == "true":
        if "true_spliced" not in layers:
            raise KeyError("missing layer 'true_spliced' for true inspection")
        return "true_spliced"
    if layer_preference == "observed":
        if "spliced" not in layers:
            raise KeyError("missing observed layer 'spliced'; use layer_preference='true' instead")
        return "spliced"
    if "spliced" in layers:
        return "spliced"
    if "true_spliced" in layers:
        return "true_spliced"
    raise KeyError("missing both 'spliced' and 'true_spliced' layers")


def _gene_index(data: Any, gene: str | int) -> int:
    names = _var_names(data)
    if isinstance(gene, int):
        if gene < 0 or gene >= len(names):
            raise IndexError(f"gene index {gene} out of range")
        return gene
    gene = str(gene)
    if gene not in names:
        raise KeyError(f"gene {gene!r} not found")
    return names.index(gene)


def _pseudotime(data: Any) -> np.ndarray:
    obs = _obs(data)
    if "pseudotime" not in obs.columns:
        raise KeyError("obs must contain pseudotime")
    return obs["pseudotime"].to_numpy(dtype=float)


def _branch(data: Any) -> np.ndarray:
    obs = _obs(data)
    if "branch" not in obs.columns:
        return np.array(["all"] * obs.shape[0], dtype=object)
    return obs["branch"].astype(str).to_numpy()


def _gene_edges(grn: Any, gene: str) -> pd.DataFrame:
    if grn is None or not hasattr(grn, "edges"):
        return pd.DataFrame()
    return grn.edges.loc[grn.edges["target"].astype(str) == str(gene)].copy()


def _incoming_targets(grn: Any, sign: str) -> set[str]:
    if grn is None or not hasattr(grn, "edges"):
        return set()
    edges = grn.edges
    return set(edges.loc[edges["sign"] == sign, "target"].astype(str))


def select_representative_genes_by_dynamics(data: Any, grn: Any | None = None) -> dict[str, Any]:
    """按 branch 后 alpha 差异自动选择代表基因。

    对 bifurcation 数据，分别在 master、activation target、repression target
    候选集合中，选择 branch_0 与 branch_1 平均 ``true_alpha`` 差异最大的
    基因。这样绘图优先展示真正有分支动态差异的基因。
    """

    genes = _var_names(data)
    incoming = set(grn.edges["target"].astype(str)) if grn is not None and hasattr(grn, "edges") else set()
    master_candidates = [gene for gene in genes if gene not in incoming] or genes[:1]
    activation_candidates = [gene for gene in genes if gene in _incoming_targets(grn, "activation")]
    repression_candidates = [gene for gene in genes if gene in _incoming_targets(grn, "repression")]

    diffs = {gene: 0.0 for gene in genes}
    try:
        alpha = _matrix(data, "true_alpha")
        branches = _branch(data)
        branch0 = branches == "branch_0"
        branch1 = branches == "branch_1"
        if branch0.any() and branch1.any():
            for idx, gene in enumerate(genes):
                diffs[gene] = float(abs(alpha[branch0, idx].mean() - alpha[branch1, idx].mean()))
    except (KeyError, ValueError):
        pass

    used: set[str] = set()

    def choose(candidates: list[str], fallback: list[str], allow_used: bool = False) -> str:
        pool = candidates or fallback or genes[:1]
        available = pool if allow_used else [gene for gene in pool if gene not in used]
        if not available:
            available = pool
        gene = max(available, key=lambda item: (diffs.get(item, 0.0), -genes.index(item)))
        used.add(gene)
        return gene

    selected = {
        "master": choose(master_candidates, genes[:1]),
        "activation_target": choose(activation_candidates, genes[min(1, len(genes) - 1): min(2, len(genes))] or genes[:1]),
        "repression_target": choose(repression_candidates, genes[min(2, len(genes) - 1): min(3, len(genes))] or genes[:1]),
    }
    return {
        "genes": selected,
        "alpha_differences": {label: diffs[gene] for label, gene in selected.items()},
        "edges": {label: _gene_edges(grn, gene) for label, gene in selected.items()},
    }


def compute_pca_embedding(
    data: Any,
    n_components: int = 2,
    layer_preference: str = "observed",
) -> tuple[np.ndarray, np.ndarray]:
    """从指定 spliced layer 计算 PCA embedding。

    ``layer_preference='observed'`` 使用 ``spliced``；``'true'`` 使用
    ``true_spliced``；``'auto'`` 优先 observed，缺失时回退到 true。返回
    ``(coords, components)``，其中 coords 是 cells x components，components
    是 components x genes，可用于把 true velocity 投影到 PCA 空间。
    """

    layer = _spliced_layer_name(data, layer_preference)
    x = _matrix(data, layer)
    if x.ndim != 2:
        raise ValueError(f"{layer} matrix must be 2D cells x genes")
    if x.shape[0] < 2:
        raise ValueError("at least two cells are required for PCA")
    n_components = min(int(n_components), x.shape[0], x.shape[1])
    if n_components < 1:
        raise ValueError("n_components must be positive")
    x_log = np.log1p(np.maximum(x, 0.0))
    centered = x_log - x_log.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:n_components].T
    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
        vt_use = np.pad(vt[:n_components], ((0, 2 - n_components), (0, 0)))
    else:
        vt_use = vt[:n_components]
    return coords[:, :2], vt_use[:2]


def compute_umap_embedding(
    data: Any,
    random_state: int = 0,
    layer_preference: str = "observed",
) -> tuple[np.ndarray, str, np.ndarray]:
    """如果安装了 umap-learn，则从 PCA 坐标继续计算 UMAP。

    UMAP 主要用于 observed layer 的定性观察。它可能在稀疏/噪声数据上
    把简单连续轨迹切碎，因此不能把 observed UMAP 当成唯一科学验证图。
    若未安装 umap-learn，会自动回退到 PCA。
    """

    pca_coords, pca_components = compute_pca_embedding(data, n_components=2, layer_preference=layer_preference)
    try:
        import umap
    except ImportError:
        return pca_coords, "pca", pca_components
    reducer = umap.UMAP(n_components=2, random_state=random_state)
    return reducer.fit_transform(pca_coords), "umap", pca_components


def _new_ax(ax=None, figsize=(6, 5)):
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _point_size(n_cells: int, emphasis: str = "default") -> float:
    if emphasis == "dense":
        return 10.0 if n_cells > 1200 else 14.0 if n_cells > 500 else 18.0
    return 14.0 if n_cells > 1200 else 18.0 if n_cells > 500 else 24.0


def _branch_color(branch: str, idx: int) -> str:
    branch = str(branch)
    if branch in DEFAULT_BRANCH_COLORS:
        return DEFAULT_BRANCH_COLORS[branch]
    cmap = plt.get_cmap("tab10")
    return cmap(idx % 10)


def _save(fig, output_path: str | Path | None):
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def _embedding_and_label(
    data: Any,
    embedding: np.ndarray | None,
    method: str,
    layer_preference: str,
    random_state: int,
) -> tuple[np.ndarray, str, np.ndarray]:
    if embedding is not None:
        _, components = compute_pca_embedding(data, layer_preference=layer_preference)
        return embedding, "embedding", components
    if method == "pca":
        coords, components = compute_pca_embedding(data, layer_preference=layer_preference)
        return coords, "pca", components
    if method == "umap":
        return compute_umap_embedding(data, random_state=random_state, layer_preference=layer_preference)
    raise ValueError("method must be 'pca' or 'umap'")


def plot_embedding_by_pseudotime(
    data: Any,
    embedding: np.ndarray | None = None,
    output_path: str | Path | None = None,
    ax=None,
    method: str = "pca",
    layer_preference: str = "observed",
    random_state: int = 0,
):
    """Scatter embedding colored by pseudotime using an explicit layer choice."""

    embedding, method_used, _ = _embedding_and_label(data, embedding, method, layer_preference, random_state)
    fig, ax = _new_ax(ax)
    pts = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=_pseudotime(data),
        cmap="viridis",
        s=_point_size(embedding.shape[0]),
        alpha=0.9,
        linewidths=0,
        rasterized=embedding.shape[0] > 800,
    )
    ax.set_title(f"{method_used.upper()} {layer_preference} by pseudotime")
    ax.set_xlabel(f"{method_used.upper()} 1")
    ax.set_ylabel(f"{method_used.upper()} 2")
    fig.colorbar(pts, ax=ax, label="pseudotime")
    return _save(fig, output_path)


def plot_embedding_by_branch(
    data: Any,
    embedding: np.ndarray | None = None,
    output_path: str | Path | None = None,
    ax=None,
    method: str = "pca",
    layer_preference: str = "observed",
    random_state: int = 0,
):
    """Scatter embedding colored by branch label using an explicit layer choice."""

    embedding, method_used, _ = _embedding_and_label(data, embedding, method, layer_preference, random_state)
    fig, ax = _new_ax(ax)
    branches = _branch(data)
    for idx, branch in enumerate(pd.unique(branches)):
        mask = branches == branch
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            label=str(branch),
            s=_point_size(mask.sum()),
            color=_branch_color(str(branch), idx),
            alpha=0.9,
            linewidths=0,
            rasterized=mask.sum() > 400,
        )
    ax.set_title(f"{method_used.upper()} {layer_preference} by branch")
    ax.set_xlabel(f"{method_used.upper()} 1")
    ax.set_ylabel(f"{method_used.upper()} 2")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, output_path)


def plot_embedding_with_velocity(
    data: Any,
    embedding: np.ndarray | None = None,
    output_path: str | Path | None = None,
    ax=None,
    stride: int | None = None,
    scale: float | None = None,
    max_arrow_length: float | None = None,
    method: str = "pca",
    layer_preference: str = "true",
    random_state: int = 0,
):
    """Scatter embedding with projected true-velocity arrows.

    The default is PCA on ``true_spliced``. Arrows are lightly subsampled and
    length-clipped for readability. PCA velocity arrows are qualitative
    quick-look diagnostics, not a full scVelo-style velocity embedding. If
    ``method='umap'`` is explicitly requested, velocity arrows are PCA-projected
    and overlaid on UMAP coordinates for qualitative visualization only.
    """

    embedding, method_used, pca_components = _embedding_and_label(data, embedding, method, layer_preference, random_state)
    velocity = _matrix(data, "true_velocity")
    projected = velocity @ pca_components.T
    if projected.shape[1] < 2:
        projected = np.pad(projected, ((0, 0), (0, 2 - projected.shape[1])))
    n_cells = embedding.shape[0]
    if stride is None:
        stride = max(1, n_cells // 35)
    if max_arrow_length is None:
        x_span = float(np.ptp(embedding[:, 0]))
        y_span = float(np.ptp(embedding[:, 1]))
        max_arrow_length = 0.08 * max((x_span**2 + y_span**2) ** 0.5, 1e-12)
    lengths = np.linalg.norm(projected[:, :2], axis=1)
    too_long = lengths > max_arrow_length
    if np.any(too_long):
        projected = projected.copy()
        projected[too_long, :2] *= (max_arrow_length / lengths[too_long])[:, None]
    idx = np.arange(0, n_cells, stride)
    fig, ax = _new_ax(ax)
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=_pseudotime(data),
        cmap="Greys",
        s=_point_size(n_cells, emphasis="dense"),
        alpha=0.45,
        linewidths=0,
        rasterized=n_cells > 800,
    )
    ax.quiver(
        embedding[idx, 0],
        embedding[idx, 1],
        projected[idx, 0],
        projected[idx, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0 if scale is None else scale,
        width=0.0026,
        color="tab:red",
        alpha=0.8,
    )
    if method_used == "umap":
        title = "UMAP with PCA-projected true velocity (qualitative only)"
    else:
        title = f"{method_used.upper()} {layer_preference} spliced with true velocity"
    ax.set_title(title)
    ax.set_xlabel(f"{method_used.upper()} 1")
    ax.set_ylabel(f"{method_used.upper()} 2")
    return _save(fig, output_path)


def plot_phase_portrait(
    data: Any,
    gene: str | int,
    output_path: str | Path | None = None,
    ax=None,
    mode: str = "true",
    use_true: bool | None = None,
    connect_by_pseudotime: bool = False,
):
    """Plot unspliced versus spliced for one gene, colored by pseudotime.

    ``mode='true'`` uses ``true_unspliced`` and ``true_spliced`` and is the
    default because observed UMI counts can be highly discrete. ``mode='observed'``
    uses ``unspliced`` and ``spliced`` and raises a clear error if they are absent.
    ``use_true`` is retained as a backward-compatible alias.
    """

    if use_true is not None:
        mode = "true" if use_true else "observed"
    if mode not in {"true", "observed"}:
        raise ValueError("mode must be 'true' or 'observed'")
    idx = _gene_index(data, gene)
    if mode == "true":
        u_layer, s_layer = "true_unspliced", "true_spliced"
    else:
        u_layer, s_layer = "unspliced", "spliced"
    u = _matrix(data, u_layer)[:, idx]
    s = _matrix(data, s_layer)[:, idx]
    branches = _branch(data)
    markers = ["o", "s", "^", "D", "P", "X"]
    fig, ax = _new_ax(ax)
    pts = None
    for b_idx, branch in enumerate(pd.unique(branches)):
        mask = branches == branch
        branch_pt = _pseudotime(data)[mask]
        pts = ax.scatter(
            u[mask],
            s[mask],
            c=branch_pt,
            cmap="viridis",
            s=_point_size(mask.sum()),
            marker=markers[b_idx % len(markers)],
            edgecolors="black",
            linewidths=0.2,
            label=str(branch),
            rasterized=mask.sum() > 400,
        )
        if connect_by_pseudotime:
            order = np.argsort(branch_pt)
            ax.plot(
                u[mask][order],
                s[mask][order],
                color=_branch_color(str(branch), b_idx),
                linewidth=0.8,
                alpha=0.5,
            )
    ax.set_title(f"Phase portrait ({mode}): {_var_names(data)[idx]}")
    ax.set_xlabel(f"{u_layer}")
    ax.set_ylabel(f"{s_layer}")
    ax.legend(frameon=False, fontsize=8)
    if pts is not None:
        fig.colorbar(pts, ax=ax, label="pseudotime")
    return _save(fig, output_path)


def plot_gene_dynamics_over_pseudotime(
    data: Any,
    gene: str | int,
    output_path: str | Path | None = None,
    ax=None,
    include_velocity_u: bool = False,
):
    """Plot true alpha, u/s, velocity, and optional velocity_u over pseudotime."""

    idx = _gene_index(data, gene)
    pt = _pseudotime(data)
    branches = _branch(data)
    quantities = {
        "true_alpha": _matrix(data, "true_alpha")[:, idx],
        "true_unspliced": _matrix(data, "true_unspliced")[:, idx],
        "true_spliced": _matrix(data, "true_spliced")[:, idx],
        "true_velocity": _matrix(data, "true_velocity")[:, idx],
    }
    if include_velocity_u:
        quantities["true_velocity_u"] = _matrix(data, "true_velocity_u")[:, idx]
    n_panels = len(quantities)
    if ax is None:
        fig, axes = plt.subplots(n_panels, 1, figsize=(7.2, 2.0 * n_panels + 0.6), sharex=True)
    else:
        fig = ax.figure
        axes = [ax]
    axes = np.atleast_1d(axes)
    for q_idx, (name, values) in enumerate(quantities.items()):
        current_ax = axes[q_idx] if len(axes) > 1 else axes[0]
        for b_idx, branch in enumerate(pd.unique(branches)):
            mask = branches == branch
            order = np.argsort(pt[mask])
            current_ax.plot(
                pt[mask][order],
                values[mask][order],
                marker="o",
                markersize=2.5,
                linewidth=1.2,
                color=_branch_color(str(branch), b_idx),
                label=str(branch),
            )
        current_ax.set_ylabel(name)
        if q_idx == 0:
            current_ax.set_title(f"Gene dynamics: {_var_names(data)[idx]}")
        if np.any(branches == "trunk"):
            trunk_max = float(pt[branches == "trunk"].max())
            current_ax.axvline(trunk_max, color="0.7", linestyle="--", linewidth=0.8)
    axes[-1].set_xlabel("pseudotime")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, frameon=False, fontsize=8)
    return _save(fig, output_path)


def plot_overview_panel(
    noisy: Any,
    lownoise: Any | None = None,
    output_path: str | Path | None = None,
    random_state: int = 0,
):
    """Create one quick-look panel for trajectory, branch, velocity, and noise views."""

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    plot_embedding_by_pseudotime(noisy, method="pca", layer_preference="true", ax=axes[0, 0])
    plot_embedding_by_branch(noisy, method="pca", layer_preference="true", ax=axes[0, 1])
    plot_embedding_with_velocity(noisy, method="pca", layer_preference="true", ax=axes[0, 2])
    plot_embedding_by_branch(noisy, method="pca", layer_preference="observed", ax=axes[1, 0])
    _, umap_method, _ = compute_umap_embedding(noisy, random_state=random_state, layer_preference="observed")
    if umap_method == "umap":
        plot_embedding_by_branch(noisy, method="umap", layer_preference="observed", random_state=random_state, ax=axes[1, 1])
    else:
        plot_embedding_by_pseudotime(noisy, method="pca", layer_preference="observed", ax=axes[1, 1])
    if lownoise is not None:
        plot_embedding_by_branch(lownoise, method="pca", layer_preference="observed", ax=axes[1, 2])
        axes[1, 2].set_title("PCA observed_lownoise by branch")
    else:
        axes[1, 2].axis("off")
    return _save(fig, output_path)


def plot_selected_gene_panel(
    data: Any,
    selected: dict[str, str],
    output_path: str | Path | None = None,
    include_velocity_u: bool = True,
):
    """Create one panel that summarizes representative genes.

    Layout: one row per selected gene, columns are true phase portrait,
    observed phase portrait, and gene dynamics.
    """

    n_rows = len(selected)
    fig = plt.figure(figsize=(16, 4.3 * n_rows), constrained_layout=True)
    outer = fig.add_gridspec(n_rows, 3, width_ratios=[1.0, 1.0, 1.35])
    for row, (label, gene) in enumerate(selected.items()):
        ax_true = fig.add_subplot(outer[row, 0])
        plot_phase_portrait(data, gene, mode="true", connect_by_pseudotime=True, ax=ax_true)
        ax_true.set_title(f"{label}: {gene} true phase")
        ax_obs = fig.add_subplot(outer[row, 1])
        try:
            plot_phase_portrait(data, gene, mode="observed", connect_by_pseudotime=False, ax=ax_obs)
            ax_obs.set_title(f"{label}: {gene} observed phase")
        except KeyError:
            ax_obs.axis("off")
        dynamic_names = ["true_alpha", "true_unspliced", "true_spliced", "true_velocity"]
        if include_velocity_u:
            dynamic_names.append("true_velocity_u")
        inner = outer[row, 2].subgridspec(len(dynamic_names), 1, hspace=0.05)
        values_map = {name: _matrix(data, name)[:, _gene_index(data, gene)] for name in dynamic_names}
        pt = _pseudotime(data)
        branches = _branch(data)
        branch_values = pd.unique(branches)
        trunk_max = float(pt[branches == "trunk"].max()) if np.any(branches == "trunk") else None
        for q_idx, name in enumerate(dynamic_names):
            ax_dyn = fig.add_subplot(inner[q_idx, 0])
            for b_idx, branch in enumerate(branch_values):
                mask = branches == branch
                order = np.argsort(pt[mask])
                ax_dyn.plot(
                    pt[mask][order],
                    values_map[name][mask][order],
                    marker="o",
                    markersize=2.2,
                    linewidth=1.0,
                    color=_branch_color(str(branch), b_idx),
                    label=str(branch) if q_idx == 0 else None,
                )
            if trunk_max is not None:
                ax_dyn.axvline(trunk_max, color="0.7", linestyle="--", linewidth=0.8)
            ax_dyn.set_ylabel(name, fontsize=8)
            if q_idx == 0:
                ax_dyn.set_title(f"{label}: {gene} dynamics")
                handles, labels = ax_dyn.get_legend_handles_labels()
                if handles:
                    ax_dyn.legend(handles, labels, frameon=False, fontsize=7, ncol=min(3, len(handles)))
            if q_idx != len(dynamic_names) - 1:
                ax_dyn.set_xticklabels([])
            else:
                ax_dyn.set_xlabel("pseudotime")
    return _save(fig, output_path)
