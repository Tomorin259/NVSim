#!/usr/bin/env python3
"""Evaluate CellOracle demo base GRN edges against Paul2015 expression with ridge regression."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse, stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from nvsim.grn import GRN
from nvsim.production import StateProductionProfile


DEFAULT_OUTDIR = ROOT / "data" / "external" / "celloracle_demo"
DEFAULT_RAW_DIR = DEFAULT_OUTDIR / "raw"
ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
GLOBAL_METRICS_FILENAME = "global_target_prediction_metrics.csv"
SKIPPED_FILENAME = "skipped_targets.csv"
CLUSTER_METRICS_FILENAME = "cluster_target_prediction_metrics.csv"
EXPORTED_GRN_FILENAME = "celloracle_paul2015_ridge_grn.csv"
MASTER_REGULATORS_FILENAME = "celloracle_paul2015_master_regulators.txt"
MR_PRODUCTION_FILENAME = "celloracle_paul2015_mr_production.csv"
SUMMARY_FILENAME = "summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--min_tfs", type=int, default=2)
    parser.add_argument("--min_target_var", type=float, default=1e-4)
    parser.add_argument("--min_cells_per_cluster", type=int, default=50)
    parser.add_argument("--min_r2", type=float, default=0.1)
    parser.add_argument("--min_pearson", type=float, default=0.3)
    parser.add_argument("--max_targets", type=int, default=500)
    parser.add_argument("--max_edges", type=int, default=3000)
    parser.add_argument("--max_tfs_per_target", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--skip_cluster_models", action="store_true")
    parser.add_argument("--export_grn_only", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Evaluate only a subset of targets for a faster smoke run.")
    parser.add_argument("--quick_n_targets", type=int, default=200)
    parser.add_argument("--use_local_raw", action="store_true", help="Load previously downloaded CellOracle raw files instead of importing celloracle.")
    parser.add_argument("--raw_dir", type=Path, default=DEFAULT_RAW_DIR)
    return parser.parse_args()


def check_runtime_dependencies(require_celloracle: bool) -> None:
    required = ["scanpy", "anndata", "pandas", "numpy", "scipy", "sklearn"]
    missing: list[str] = []
    for module_name in required:
        try:
            __import__(module_name)
        except Exception:
            missing.append(module_name)
    if missing:
        raise RuntimeError(f"Missing required Python packages: {', '.join(sorted(missing))}")
    if require_celloracle:
        try:
            __import__("celloracle")
        except Exception as exc:
            raise RuntimeError(
                "celloracle is required for the default loader path but is not available. "
                "Either install a dedicated environment such as:\n"
                "  conda create -n celloracle python=3.8\n"
                "  conda activate celloracle\n"
                "  pip install celloracle scanpy scikit-learn\n"
                "or rerun this script with --use_local_raw after downloading the demo files."
            ) from exc


def load_demo_inputs(args: argparse.Namespace):
    import anndata as ad

    if args.use_local_raw:
        h5ad_path = args.raw_dir / "Paul_etal_v202204.h5ad"
        grn_path = args.raw_dir / "mm9_mouse_atac_atlas_data_TSS_and_cicero_0.9_accum_threshold_10.5_DF_peaks_by_TFs_v202204.parquet"
        if not h5ad_path.exists():
            raise FileNotFoundError(f"Missing local Paul2015 h5ad file: {h5ad_path}")
        if not grn_path.exists():
            raise FileNotFoundError(f"Missing local base GRN parquet file: {grn_path}")
        adata = ad.read_h5ad(h5ad_path)
        base_grn = pd.read_parquet(grn_path)
        data_source = {
            "adata": str(h5ad_path),
            "base_grn": str(grn_path),
            "loader": "local_raw",
        }
        return adata, base_grn, data_source

    import celloracle as co

    adata = co.data.load_Paul2015_data()
    base_grn = co.data.load_mouse_scATAC_atlas_base_GRN()
    data_source = {
        "adata": "co.data.load_Paul2015_data()",
        "base_grn": "co.data.load_mouse_scATAC_atlas_base_GRN()",
        "loader": "celloracle",
    }
    return adata, base_grn, data_source


def choose_expression_source(adata):
    if "raw_count" in adata.layers:
        return adata.layers["raw_count"], "raw_count"
    return adata.X, "X"


def choose_cluster_column(obs: pd.DataFrame) -> str | None:
    for candidate in ["louvain_annot", "cell_type", "louvain"]:
        if candidate in obs.columns:
            return candidate
    return None


def preprocess_expression(adata) -> tuple[pd.DataFrame, str]:
    import scanpy as sc

    matrix, layer_name = choose_expression_source(adata)
    work = adata.copy()
    work.X = matrix.copy() if hasattr(matrix, "copy") else matrix
    sc.pp.normalize_total(work, target_sum=1e4)
    sc.pp.log1p(work)
    expr = work.X.toarray() if sparse.issparse(work.X) else np.asarray(work.X)
    expression_df = pd.DataFrame(expr, index=work.obs_names.astype(str), columns=work.var_names.astype(str))
    return expression_df, layer_name


def _is_non_tf_column(column_name: str, target_column: str) -> bool:
    lowered = column_name.lower()
    return column_name == target_column or lowered in {
        "peak_id",
        "peak",
        "peak_name",
        "gene_short_name",
        "gene_name",
        "gene",
        "target",
        "target_gene",
    }


def extract_target_to_tfs(base_grn: pd.DataFrame, expressed_genes: Iterable[str]) -> tuple[dict[str, list[str]], pd.DataFrame, dict[str, object]]:
    expressed = {str(gene) for gene in expressed_genes}
    if "gene_short_name" in base_grn.columns:
        target_column = "gene_short_name"
    else:
        candidate_columns = [c for c in base_grn.columns if "gene" in str(c).lower() or "target" in str(c).lower()]
        if not candidate_columns:
            raise ValueError("Could not identify target column in CellOracle base GRN.")
        target_column = candidate_columns[0]

    tf_columns = [column for column in base_grn.columns if not _is_non_tf_column(str(column), target_column)]
    tf_columns = [column for column in tf_columns if str(column) in expressed]

    target_to_tfs: dict[str, set[str]] = {}
    target_counter = Counter()
    tf_counter = Counter()

    target_values = base_grn[target_column].astype(str).to_numpy()
    tf_matrix = base_grn.loc[:, tf_columns].fillna(0).to_numpy(dtype=np.int8)
    tf_names = np.asarray([str(column) for column in tf_columns])

    for row_index, target in enumerate(target_values):
        if target not in expressed:
            continue
        active_index = np.flatnonzero(tf_matrix[row_index] > 0)
        if active_index.size == 0:
            continue
        active_tfs = [tf for tf in tf_names[active_index] if tf != target]
        if not active_tfs:
            continue
        target_to_tfs.setdefault(target, set()).update(active_tfs)

    edge_records: list[dict[str, object]] = []
    for target, tf_set in sorted(target_to_tfs.items()):
        for tf in sorted(tf_set):
            edge_records.append({"source": tf, "target": target, "base_grn_value": 1})
            target_counter[target] += 1
            tf_counter[tf] += 1

    base_edges = pd.DataFrame.from_records(edge_records, columns=["source", "target", "base_grn_value"])
    upstream_sizes = [len(tf_set) for tf_set in target_to_tfs.values()]
    stats = {
        "base_grn_shape": list(base_grn.shape),
        "target_column": target_column,
        "n_raw_targets": int(base_grn[target_column].astype(str).nunique()),
        "n_candidate_targets": int(len(target_to_tfs)),
        "n_candidate_tfs": int(len(tf_counter)),
        "n_candidate_edges": int(len(edge_records)),
        "upstream_tf_distribution": {
            "min": int(min(upstream_sizes)) if upstream_sizes else 0,
            "median": float(np.median(upstream_sizes)) if upstream_sizes else 0.0,
            "mean": float(np.mean(upstream_sizes)) if upstream_sizes else 0.0,
            "max": int(max(upstream_sizes)) if upstream_sizes else 0,
        },
    }
    return {target: sorted(tf_set) for target, tf_set in target_to_tfs.items()}, base_edges, stats


def _can_stratify(labels: pd.Series | None) -> bool:
    if labels is None:
        return False
    counts = labels.astype(str).value_counts()
    return not counts.empty and bool((counts >= 2).all()) and counts.shape[0] >= 2


def _safe_corr(values_a: np.ndarray, values_b: np.ndarray, method: str) -> float:
    if np.allclose(values_a, values_a[0]) or np.allclose(values_b, values_b[0]):
        return float("nan")
    if method == "pearson":
        return float(np.corrcoef(values_a, values_b)[0, 1])
    if method == "spearman":
        return float(stats.spearmanr(values_a, values_b).statistic)
    raise ValueError(f"Unsupported correlation method: {method}")


def _serialize_float_list(values: Iterable[float]) -> str:
    return ";".join(f"{float(value):.12g}" for value in values)


def _fit_ridge_for_target(
    target: str,
    expression_df: pd.DataFrame,
    target_to_tfs: dict[str, list[str]],
    min_tfs: int,
    min_target_var: float,
    test_size: float,
    seed: int,
    stratify_labels: pd.Series | None,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    tf_list = list(target_to_tfs[target])
    if len(tf_list) < min_tfs:
        return None, {"target": target, "reason": "too_few_candidate_tfs", "n_tfs": len(tf_list)}

    y = expression_df[target].to_numpy(dtype=float)
    target_var = float(np.var(y))
    if not np.isfinite(target_var) or target_var <= min_target_var:
        return None, {"target": target, "reason": "low_target_variance", "n_tfs": len(tf_list), "target_var": target_var}

    X = expression_df.loc[:, tf_list].to_numpy(dtype=float)
    feature_var = np.var(X, axis=0)
    keep_mask = np.isfinite(feature_var) & (feature_var > 1e-8)
    if keep_mask.sum() < min_tfs:
        return None, {"target": target, "reason": "low_tf_variance", "n_tfs": int(keep_mask.sum()), "target_var": target_var}

    tf_array = np.asarray(tf_list)[keep_mask]
    X = X[:, keep_mask]

    split_kwargs: dict[str, object] = {"test_size": test_size, "random_state": seed}
    if _can_stratify(stratify_labels):
        split_kwargs["stratify"] = stratify_labels.astype(str)
        X_train, X_test, y_train, y_test, _, label_test = train_test_split(X, y, stratify_labels.astype(str).to_numpy(), **split_kwargs)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)
        label_test = None

    if np.var(y_train) <= min_target_var:
        return None, {"target": target, "reason": "low_train_variance", "n_tfs": int(X.shape[1]), "target_var": target_var}

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RidgeCV(alphas=ALPHA_GRID)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    with np.errstate(invalid="ignore"):
        coef = model.coef_ / np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    coef = np.asarray(coef, dtype=float)

    metrics = {
        "target": target,
        "r2": float(r2_score(y_test, y_pred)),
        "pearson": _safe_corr(y_test, y_pred, method="pearson"),
        "spearman": _safe_corr(y_test, y_pred, method="spearman"),
        "mse": float(mean_squared_error(y_test, y_pred)),
        "n_tfs": int(X.shape[1]),
        "n_cells": int(X.shape[0]),
        "best_alpha": float(model.alpha_),
        "target_mean": float(np.mean(y)),
        "target_var": target_var,
        "tfs": ";".join(tf_array.tolist()),
        "coefs": _serialize_float_list(coef.tolist()),
        "cluster_labels_used": bool(stratify_labels is not None and _can_stratify(stratify_labels)),
    }
    if label_test is not None:
        metrics["n_test_clusters"] = int(pd.Series(label_test).nunique())
    return metrics, None


def fit_global_ridge_models(
    expression_df: pd.DataFrame,
    target_to_tfs: dict[str, list[str]],
    cluster_labels: pd.Series | None,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered_targets = sorted(target_to_tfs, key=lambda target: (-len(target_to_tfs[target]), target))
    if args.quick:
        ordered_targets = ordered_targets[: min(args.quick_n_targets, len(ordered_targets))]

    def worker(target: str):
        return _fit_ridge_for_target(
            target=target,
            expression_df=expression_df,
            target_to_tfs=target_to_tfs,
            min_tfs=args.min_tfs,
            min_target_var=args.min_target_var,
            test_size=args.test_size,
            seed=args.seed,
            stratify_labels=cluster_labels,
        )

    results = (
        Parallel(n_jobs=args.n_jobs, prefer="threads")(delayed(worker)(target) for target in ordered_targets)
        if args.n_jobs != 1
        else [worker(target) for target in ordered_targets]
    )

    metrics_records = [metrics for metrics, skipped in results if metrics is not None]
    skipped_records = [skipped for metrics, skipped in results if skipped is not None]
    metrics_df = pd.DataFrame(metrics_records).sort_values(["r2", "pearson", "target"], ascending=[False, False, True]).reset_index(drop=True)
    if skipped_records:
        skipped_df = pd.DataFrame(skipped_records).sort_values(["reason", "target"]).reset_index(drop=True)
    else:
        skipped_df = pd.DataFrame(columns=["target", "reason"])
    return metrics_df, skipped_df


def fit_cluster_specific_ridge_models(
    expression_df: pd.DataFrame,
    target_to_tfs: dict[str, list[str]],
    cluster_labels: pd.Series | None,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if cluster_labels is None:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for cluster_name, cluster_index in cluster_labels.astype(str).groupby(cluster_labels.astype(str)).groups.items():
        if len(cluster_index) < args.min_cells_per_cluster:
            continue
        expr_subset = expression_df.loc[cluster_index]
        ordered_targets = sorted(target_to_tfs, key=lambda target: (-len(target_to_tfs[target]), target))
        if args.quick:
            ordered_targets = ordered_targets[: min(args.quick_n_targets, len(ordered_targets))]
        def worker(target: str):
            return _fit_ridge_for_target(
                target=target,
                expression_df=expr_subset,
                target_to_tfs=target_to_tfs,
                min_tfs=args.min_tfs,
                min_target_var=args.min_target_var,
                test_size=args.test_size,
                seed=args.seed,
                stratify_labels=None,
            )
        raw_results = (
            Parallel(n_jobs=args.n_jobs, prefer="threads")(delayed(worker)(target) for target in ordered_targets)
            if args.n_jobs != 1
            else [worker(target) for target in ordered_targets]
        )
        results = []
        for metrics, _ in raw_results:
            if metrics is None:
                continue
            metrics["cluster"] = str(cluster_name)
            results.append(metrics)
        if results:
            frames.append(pd.DataFrame(results))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["cluster", "r2", "pearson", "target"], ascending=[True, False, False, True]).reset_index(drop=True)


def _parse_serialized_values(serialized: str, cast) -> list:
    if not isinstance(serialized, str) or not serialized:
        return []
    return [cast(token) for token in serialized.split(";") if token]


def _scale_edge_strengths(abs_coefs: np.ndarray) -> np.ndarray:
    if abs_coefs.size == 0:
        return abs_coefs
    min_value = float(abs_coefs.min())
    max_value = float(abs_coefs.max())
    if not np.isfinite(min_value) or not np.isfinite(max_value) or math.isclose(min_value, max_value):
        return np.full(abs_coefs.shape, 1.0, dtype=float)
    scaled = 0.2 + 1.8 * (abs_coefs - min_value) / (max_value - min_value)
    return scaled.astype(float)


def export_expression_supported_grn(metrics_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame(columns=[
            "regulator",
            "target",
            "sign",
            "K",
            "half_response",
            "hill_coefficient",
            "source",
            "evidence",
            "r2",
            "pearson",
            "spearman",
            "coef",
        ])

    passing = metrics_df.loc[
        (
            metrics_df["r2"].fillna(-np.inf) >= args.min_r2
        ) | (
            metrics_df["pearson"].fillna(-np.inf) >= args.min_pearson
        )
    ].copy()
    passing = passing.loc[(passing["n_tfs"] >= args.min_tfs) & (passing["target_var"] > args.min_target_var)].copy()
    passing = passing.sort_values(["r2", "pearson", "target"], ascending=[False, False, True]).head(args.max_targets)

    edge_rows: list[dict[str, object]] = []
    for row in passing.itertuples(index=False):
        tf_list = _parse_serialized_values(row.tfs, str)
        coef_list = np.asarray(_parse_serialized_values(row.coefs, float), dtype=float)
        if len(tf_list) != len(coef_list):
            continue
        order = np.argsort(-np.abs(coef_list))[: args.max_tfs_per_target]
        for rank in order:
            coef = float(coef_list[rank])
            tf = str(tf_list[rank])
            edge_rows.append(
                {
                    "regulator": tf,
                    "target": str(row.target),
                    "coef": coef,
                    "abs_coef": abs(coef),
                    "sign": "+" if coef >= 0 else "-",
                    "half_response": 1.0,
                    "hill_coefficient": 2.0,
                    "source": "CellOracle_Paul2015_baseGRN_Ridge",
                    "evidence": "baseGRN_plus_expression_fit",
                    "r2": float(row.r2),
                    "pearson": float(row.pearson) if pd.notna(row.pearson) else float("nan"),
                    "spearman": float(row.spearman) if pd.notna(row.spearman) else float("nan"),
                }
            )

    if not edge_rows:
        return pd.DataFrame(columns=[
            "regulator",
            "target",
            "sign",
            "K",
            "half_response",
            "hill_coefficient",
            "source",
            "evidence",
            "r2",
            "pearson",
            "spearman",
            "coef",
        ])

    edges_df = pd.DataFrame(edge_rows)
    edges_df = edges_df.sort_values(["r2", "pearson", "abs_coef", "target", "regulator"], ascending=[False, False, False, True, True]).head(args.max_edges).reset_index(drop=True)
    edges_df["K"] = _scale_edge_strengths(edges_df["abs_coef"].to_numpy(dtype=float))
    exported = edges_df.loc[:, [
        "regulator",
        "target",
        "sign",
        "K",
        "half_response",
        "hill_coefficient",
        "source",
        "evidence",
        "r2",
        "pearson",
        "spearman",
        "coef",
    ]].copy()
    return exported


def choose_master_regulators(exported_grn: pd.DataFrame, n_masters: int = 10) -> list[str]:
    if exported_grn.empty:
        return []
    out_degree = Counter(exported_grn["regulator"].astype(str))
    in_degree = Counter(exported_grn["target"].astype(str))
    tf_candidates = sorted(out_degree, key=lambda tf: (-out_degree[tf], in_degree.get(tf, 0), tf))
    zero_indegree = [tf for tf in tf_candidates if in_degree.get(tf, 0) == 0]
    selected = zero_indegree[: min(n_masters, len(zero_indegree))]
    if len(selected) < min(n_masters, len(tf_candidates)):
        for tf in tf_candidates:
            if tf in selected:
                continue
            selected.append(tf)
            if len(selected) >= min(n_masters, len(tf_candidates)):
                break
    return selected[: min(n_masters, len(tf_candidates))]


def _find_lineage_masks(obs: pd.DataFrame, cluster_column: str | None) -> dict[str, np.ndarray]:
    n_cells = obs.shape[0]
    all_true = np.ones(n_cells, dtype=bool)
    root_mask = all_true.copy()
    branch0_mask = all_true.copy()
    branch1_mask = all_true.copy()

    if "dpt_pseudotime" in obs.columns:
        pseudotime = pd.to_numeric(obs["dpt_pseudotime"], errors="coerce")
        threshold = np.nanquantile(pseudotime, 0.2)
        root_mask = np.asarray(pseudotime <= threshold)

    if cluster_column is not None:
        labels = obs[cluster_column].astype(str)
        lower = labels.str.lower()
        erythroid_mask = lower.str.contains("ery")
        myeloid_mask = lower.str.contains("gran|gmp|gmpl|mo|mono|neu")
        mep_mask = lower.str.contains("mep")
        if erythroid_mask.any():
            branch0_mask = erythroid_mask.to_numpy()
        if myeloid_mask.any():
            branch1_mask = myeloid_mask.to_numpy()
        if mep_mask.any():
            root_mask = mep_mask.to_numpy()

    return {"root": root_mask, "branch0": branch0_mask, "branch1": branch1_mask}


def make_master_production_profile(
    expression_df: pd.DataFrame,
    obs: pd.DataFrame,
    master_regulators: list[str],
    cluster_column: str | None,
    seed: int,
) -> pd.DataFrame:
    if not master_regulators:
        return pd.DataFrame(index=pd.Index(["root", "branch0", "branch1"], name="state"))

    masks = _find_lineage_masks(obs, cluster_column)
    state_means = []
    rng = np.random.default_rng(seed)
    for state in ["root", "branch0", "branch1"]:
        mask = masks[state]
        if mask.sum() == 0:
            values = expression_df.loc[:, master_regulators].mean(axis=0).to_numpy(dtype=float)
            values = values + rng.uniform(-0.05, 0.05, size=len(master_regulators))
        else:
            values = expression_df.loc[mask, master_regulators].mean(axis=0).to_numpy(dtype=float)
        state_means.append(values)

    matrix = np.vstack(state_means).astype(float)
    finite = np.isfinite(matrix)
    if not finite.any() or np.nanmax(matrix) <= 0:
        matrix = np.full(matrix.shape, 1.0, dtype=float)
    else:
        scale = float(np.nanpercentile(matrix[finite], 90))
        scale = scale if scale > 0 else 1.0
        matrix = np.clip(matrix / scale * 2.0, 0.3, 3.0)
        if np.allclose(matrix[1], matrix[2]):
            matrix[2] = np.clip(matrix[2] * 0.9 + 0.1, 0.3, 3.0)

    return pd.DataFrame(matrix, index=pd.Index(["root", "branch0", "branch1"], name="state"), columns=master_regulators).round(6)


def write_master_regulator_files(outdir: Path, master_regulators: list[str], production_profile: pd.DataFrame) -> None:
    (outdir / MASTER_REGULATORS_FILENAME).write_text("\n".join(master_regulators) + ("\n" if master_regulators else ""), encoding="utf-8")
    production_profile.reset_index().to_csv(outdir / MR_PRODUCTION_FILENAME, index=False)


def make_figures(outdir: Path, metrics_df: pd.DataFrame, exported_grn: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    figures_dir = outdir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(metrics_df["r2"].dropna(), bins=30, color="#4c78a8")
        ax.set_title("Global target prediction R2")
        ax.set_xlabel("R2")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(figures_dir / "r2_histogram.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(metrics_df["pearson"].dropna(), bins=30, color="#f58518")
        ax.set_title("Global target prediction Pearson")
        ax.set_xlabel("Pearson")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(figures_dir / "pearson_histogram.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(metrics_df["n_tfs"].dropna(), bins=30, color="#54a24b")
        ax.set_title("Upstream TFs per modeled target")
        ax.set_xlabel("n_tfs")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(figures_dir / "n_tfs_per_target_histogram.png", dpi=150)
        plt.close(fig)

        top_targets = metrics_df.sort_values(["r2", "pearson"], ascending=[False, False]).head(15)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(top_targets["target"], top_targets["r2"], color="#e45756")
        ax.invert_yaxis()
        ax.set_xlabel("R2")
        ax.set_title("Top predicted targets")
        fig.tight_layout()
        fig.savefig(figures_dir / "top_targets_barplot.png", dpi=150)
        plt.close(fig)

    if not exported_grn.empty:
        out_degree = exported_grn["regulator"].astype(str).value_counts()
        in_degree = exported_grn["target"].astype(str).value_counts()
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].hist(out_degree.to_numpy(), bins=20, color="#72b7b2")
        axes[0].set_title("Exported GRN out-degree")
        axes[0].set_xlabel("Edges per TF")
        axes[0].set_ylabel("Count")
        axes[1].hist(in_degree.to_numpy(), bins=20, color="#b279a2")
        axes[1].set_title("Exported GRN in-degree")
        axes[1].set_xlabel("Edges per target")
        axes[1].set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(figures_dir / "degree_distribution_exported_grn.png", dpi=150)
        plt.close(fig)


def validate_exported_artifacts(exported_grn: pd.DataFrame, production_profile: pd.DataFrame, master_regulators: list[str]) -> None:
    if exported_grn.empty:
        return
    genes = sorted(set(exported_grn["regulator"].astype(str)).union(exported_grn["target"].astype(str)))
    GRN.from_dataframe(exported_grn.loc[:, ["regulator", "target", "sign", "K", "half_response", "hill_coefficient"]], genes=genes, master_regulators=master_regulators)
    StateProductionProfile(production_profile)


def build_summary(
    adata,
    expression_layer_used: str,
    cluster_column_used: str | None,
    base_stats: dict[str, object],
    metrics_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
    exported_grn: pd.DataFrame,
    master_regulators: list[str],
    output_files: dict[str, str],
    quick_mode: bool,
    skip_cluster_models: bool,
) -> dict[str, object]:
    summary = {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "cluster_column_used": cluster_column_used,
        "expression_layer_used": expression_layer_used,
        "quick_mode": bool(quick_mode),
        "skip_cluster_models": bool(skip_cluster_models),
        "base_grn_shape": base_stats["base_grn_shape"],
        "n_candidate_edges": int(base_stats["n_candidate_edges"]),
        "n_candidate_targets": int(base_stats["n_candidate_targets"]),
        "n_candidate_tfs": int(base_stats["n_candidate_tfs"]),
        "n_modeled_targets": int(metrics_df.shape[0]),
        "n_skipped_targets": int(skipped_df.shape[0]),
        "median_r2": float(metrics_df["r2"].median()) if not metrics_df.empty else float("nan"),
        "median_pearson": float(metrics_df["pearson"].median()) if not metrics_df.empty else float("nan"),
        "number_of_targets_passing_filter": int(exported_grn["target"].astype(str).nunique()) if not exported_grn.empty else 0,
        "exported_grn_edges": int(exported_grn.shape[0]),
        "exported_grn_genes": int(len(set(exported_grn["regulator"].astype(str)).union(exported_grn["target"].astype(str)))) if not exported_grn.empty else 0,
        "exported_grn_tfs": int(exported_grn["regulator"].astype(str).nunique()) if not exported_grn.empty else 0,
        "master_regulators": master_regulators,
        "output_files": output_files,
    }
    return summary


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    check_runtime_dependencies(require_celloracle=not args.use_local_raw)

    if args.export_grn_only:
        metrics_df = pd.read_csv(args.outdir / GLOBAL_METRICS_FILENAME)
        adata, _, _ = load_demo_inputs(args)
        expression_df, expression_layer_used = preprocess_expression(adata)
        cluster_column = choose_cluster_column(adata.obs)
        exported_grn = export_expression_supported_grn(metrics_df, args)
        master_regulators = choose_master_regulators(exported_grn)
        production_profile = make_master_production_profile(expression_df, adata.obs, master_regulators, cluster_column, args.seed)
        exported_grn.to_csv(args.outdir / EXPORTED_GRN_FILENAME, index=False)
        write_master_regulator_files(args.outdir, master_regulators, production_profile)
        validate_exported_artifacts(exported_grn, production_profile, master_regulators)
        return

    adata, base_grn, data_source = load_demo_inputs(args)
    print("adata.shape", adata.shape)
    print("adata.obs columns", list(adata.obs.columns))
    print("adata.layers keys", list(adata.layers.keys()))
    print("adata.obsm keys", list(adata.obsm.keys()))
    print("base_GRN.shape", base_grn.shape)
    print("base_GRN columns", list(base_grn.columns[:12]))
    print(base_grn.iloc[:2, :12].to_string())

    expression_df, expression_layer_used = preprocess_expression(adata)
    cluster_column = choose_cluster_column(adata.obs)
    cluster_labels = adata.obs[cluster_column].astype(str) if cluster_column is not None else None

    target_to_tfs, base_edges, base_stats = extract_target_to_tfs(base_grn, expression_df.columns)
    global_metrics_df, skipped_df = fit_global_ridge_models(expression_df, target_to_tfs, cluster_labels, args)
    cluster_metrics_df = pd.DataFrame() if args.skip_cluster_models else fit_cluster_specific_ridge_models(expression_df, target_to_tfs, cluster_labels, args)
    exported_grn = export_expression_supported_grn(global_metrics_df, args)
    master_regulators = choose_master_regulators(exported_grn)
    production_profile = make_master_production_profile(expression_df, adata.obs, master_regulators, cluster_column, args.seed)

    global_metrics_path = args.outdir / GLOBAL_METRICS_FILENAME
    skipped_path = args.outdir / SKIPPED_FILENAME
    cluster_metrics_path = args.outdir / CLUSTER_METRICS_FILENAME
    exported_grn_path = args.outdir / EXPORTED_GRN_FILENAME
    base_edges_path = args.outdir / "base_edges.csv"

    global_metrics_df.to_csv(global_metrics_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)
    if not cluster_metrics_df.empty:
        cluster_metrics_df.to_csv(cluster_metrics_path, index=False)
    elif cluster_metrics_path.exists():
        cluster_metrics_path.unlink()
    exported_grn.to_csv(exported_grn_path, index=False)
    base_edges.to_csv(base_edges_path, index=False)
    write_master_regulator_files(args.outdir, master_regulators, production_profile)
    validate_exported_artifacts(exported_grn, production_profile, master_regulators)
    make_figures(args.outdir, global_metrics_df, exported_grn)

    output_files = {
        "global_metrics_csv": str(global_metrics_path),
        "skipped_targets_csv": str(skipped_path),
        "cluster_metrics_csv": str(cluster_metrics_path) if not cluster_metrics_df.empty else None,
        "exported_grn_csv": str(exported_grn_path),
        "base_edges_csv": str(base_edges_path),
        "master_regulators_txt": str(args.outdir / MASTER_REGULATORS_FILENAME),
        "mr_production_csv": str(args.outdir / MR_PRODUCTION_FILENAME),
        "figures_dir": str(args.outdir / "figures"),
        "data_source": data_source,
    }
    summary = build_summary(
        adata,
        expression_layer_used,
        cluster_column,
        base_stats,
        global_metrics_df,
        skipped_df,
        exported_grn,
        master_regulators,
        output_files,
        quick_mode=args.quick,
        skip_cluster_models=args.skip_cluster_models,
    )
    summary_path = args.outdir / SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
