"""NVSim 的 GRN（gene regulatory network）表示与校验。

本模块只负责“网络是什么”，不负责真正计算动力学。
NVSim 当前把 SERGIO-style GRN 参数标准化成：

- regulator
- target
- sign
- K
- half_response
- hill_coefficient

为了兼容旧接口，``weight`` 作为 ``K`` 的别名、``threshold`` 作为
``half_response`` 的别名仍然接受，但标准化后的 GRN 内部只保留
canonical 列名。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import warnings

import numpy as np
import pandas as pd

from .config import GRNConfig

REQUIRED_COLUMNS = ("regulator", "target", "sign")
OPTIONAL_COLUMNS = ("K", "weight", "hill_coefficient", "half_response", "threshold")
RETURN_COLUMNS = ("regulator", "target", "sign", "K", "half_response", "hill_coefficient")
VALID_SIGNS = {"activation", "repression"}
LEGACY_GRN_COLUMN_ALIASES = {"weight": "K", "threshold": "half_response"}
HALF_RESPONSE_CALIBRATIONS = {"off", "auto", "topology_propagation", "cyclic"}
_SIGN_ALIASES = {
    "activation": "activation",
    "activating": "activation",
    "act": "activation",
    "+": "activation",
    "+1": "activation",
    "1": "activation",
    "repression": "repression",
    "repressive": "repression",
    "rep": "repression",
    "-": "repression",
    "-1": "repression",
}


def normalize_sign(value: object) -> str:
    """把用户常见的边符号统一成 activation 或 repression。

例如 ``+``、``act`` 会被标准化成 ``activation``；``-``、``rep``
会被标准化成 ``repression``。这样后续计算不需要处理多种写法。
"""

    key = str(value).strip().lower()
    if key not in _SIGN_ALIASES:
        raise ValueError(f"invalid edge sign {value!r}; expected activation or repression")
    return _SIGN_ALIASES[key]


def _coerce_mean_expression(
    regulator_expression: pd.Series | pd.DataFrame | dict[str, float],
) -> pd.Series:
    if isinstance(regulator_expression, pd.Series):
        means = regulator_expression.astype(float).copy()
    elif isinstance(regulator_expression, pd.DataFrame):
        if regulator_expression.empty:
            raise ValueError("regulator_expression dataframe must not be empty")
        if regulator_expression.columns.size > 0:
            means = regulator_expression.astype(float).mean(axis=0)
        else:
            raise ValueError("regulator_expression dataframe must contain gene columns")
    else:
        means = pd.Series(regulator_expression, dtype=float)
    means.index = means.index.astype(str)
    values = means.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("regulator mean expression must be finite")
    if (values < 0).any():
        raise ValueError("regulator mean expression must be non-negative")
    return means


def validate_grn(edges: pd.DataFrame, config: GRNConfig | None = None) -> pd.DataFrame:
    """校验并标准化 GRN 边表。

    输入的 ``edges`` 是 pandas DataFrame。这里会检查必需列、缺失值、
    权重非负、Hill 参数为正，并把 sign 统一成标准字符串。

    返回值仍然是 DataFrame，但列顺序固定，且只保留 canonical 列名。
    ``weight`` / ``threshold`` 仅作为输入兼容别名接受。若
    ``half_response`` 缺失，则保留为 ``NaN``，留给后续校准流程填充。
    """

    config = config or GRNConfig()
    missing = [col for col in REQUIRED_COLUMNS if col not in edges.columns]
    if missing:
        raise ValueError(f"GRN is missing required columns: {missing}")
    if "K" not in edges.columns and "weight" not in edges.columns:
        raise ValueError("GRN must contain either 'K' or 'weight'")
    for legacy_name, canonical_name in LEGACY_GRN_COLUMN_ALIASES.items():
        if legacy_name in edges.columns and canonical_name not in edges.columns:
            warnings.warn(
                f"GRN column {legacy_name!r} is a legacy alias for {canonical_name!r}; prefer {canonical_name!r}.",
                DeprecationWarning,
                stacklevel=2,
            )

    normalized = edges.copy()
    for col in ("regulator", "target"):
        if normalized[col].isna().any():
            raise ValueError(f"GRN column {col!r} contains missing values")
        normalized[col] = normalized[col].astype(str)

    normalized["sign"] = normalized["sign"].map(normalize_sign)

    if "K" in normalized.columns:
        normalized["K"] = pd.to_numeric(normalized["K"], errors="raise")
    if "weight" in normalized.columns:
        normalized["weight"] = pd.to_numeric(normalized["weight"], errors="raise")
    if "K" not in normalized.columns:
        normalized["K"] = normalized["weight"]

    K_values = normalized["K"].to_numpy(dtype=float)
    if "weight" in normalized.columns:
        weight_values = normalized["weight"].to_numpy(dtype=float)
        if not np.isfinite(weight_values).all():
            raise ValueError("GRN K/weight values must be finite")
        if (weight_values < 0).any():
            raise ValueError("GRN K/weight values must be non-negative; sign controls activation/repression")
        if not np.allclose(K_values, weight_values):
            raise ValueError("GRN columns 'K' and 'weight' must match when both are provided")
    if not np.isfinite(K_values).all():
        raise ValueError("GRN K values must be finite")
    if (K_values < 0).any():
        raise ValueError("GRN K values must be non-negative; sign controls activation/repression")

    if "hill_coefficient" not in normalized.columns:
        normalized["hill_coefficient"] = config.default_hill_coefficient
    if "half_response" not in normalized.columns and "threshold" in normalized.columns:
        normalized["half_response"] = normalized["threshold"]
    if "half_response" not in normalized.columns:
        normalized["half_response"] = np.nan

    for col in OPTIONAL_COLUMNS:
        if col not in normalized.columns:
            continue
        normalized[col] = pd.to_numeric(normalized[col], errors="raise")
        values = normalized[col].to_numpy(dtype=float)
        if col == "K" or col == "weight":
            if not np.isfinite(values).all():
                raise ValueError(f"GRN column {col!r} must be finite")
            if (values < 0).any():
                raise ValueError(f"GRN column {col!r} must be non-negative")
        elif col == "hill_coefficient":
            if not np.isfinite(values).all():
                raise ValueError(f"GRN column {col!r} must be finite")
            if (values <= 0).any():
                raise ValueError(f"GRN column {col!r} must be positive")
        else:
            finite_mask = ~np.isnan(values)
            if not np.isfinite(values[finite_mask]).all():
                raise ValueError(f"GRN column {col!r} must be finite when provided")
            if (values[finite_mask] <= 0).any():
                raise ValueError(f"GRN column {col!r} must be positive")

    return normalized[list(RETURN_COLUMNS)]


def identify_master_regulators(
    grn: GRN | pd.DataFrame,
    explicit_master_regulators: Iterable[str] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return master regulators and non-master genes.

    Priority:
    1. explicit_master_regulators
    2. GRN.master_regulators
    3. no-incoming-edge inference
    """

    if isinstance(grn, GRN):
        edges = grn.edges
        genes = grn.genes
        stored_masters = grn.master_regulators
    else:
        edges = validate_grn(grn)
        genes = tuple(sorted(set(edges["regulator"]).union(edges["target"])))
        stored_masters = None

    if explicit_master_regulators is not None:
        masters = tuple(str(gene) for gene in explicit_master_regulators)
    elif stored_masters is not None:
        masters = tuple(str(gene) for gene in stored_masters)
    else:
        incoming = set(edges["target"].astype(str))
        masters = tuple(str(gene) for gene in genes if str(gene) not in incoming)

    missing = sorted(set(masters) - set(map(str, genes)))
    if missing:
        raise ValueError(f"master regulators absent from gene list: {missing}")
    master_set = set(masters)
    non_masters = tuple(str(gene) for gene in genes if str(gene) not in master_set)
    return masters, non_masters


def build_graph_levels(
    grn: GRN | pd.DataFrame,
    explicit_master_regulators: Iterable[str] | None = None,
) -> dict[str, object]:
    """Build SERGIO-style level metadata without requiring acyclicity.

    For acyclic GRNs, returns exact ``gene_to_level`` with master regulators at
    level 0. For cyclic GRNs, returns partial level metadata and warnings but
    does not fail.
    """

    if isinstance(grn, GRN):
        edges = grn.edges
        genes = tuple(str(gene) for gene in grn.genes)
    else:
        edges = validate_grn(grn)
        genes = tuple(sorted(set(edges["regulator"].astype(str)).union(edges["target"].astype(str))))

    masters, _ = identify_master_regulators(grn, explicit_master_regulators)
    incoming: dict[str, set[str]] = {gene: set() for gene in genes}
    outgoing: dict[str, set[str]] = {gene: set() for gene in genes}
    autoregulated: list[str] = []
    for edge in edges.itertuples(index=False):
        regulator = str(edge.regulator)
        target = str(edge.target)
        outgoing[regulator].add(target)
        incoming[target].add(regulator)
        if regulator == target:
            autoregulated.append(regulator)

    visited: set[str] = set()
    active: set[str] = set()
    cycle_nodes: set[str] = set()

    def dfs(node: str) -> None:
        visited.add(node)
        active.add(node)
        for nxt in outgoing[node]:
            if nxt not in visited:
                dfs(nxt)
            elif nxt in active:
                cycle_nodes.update({node, nxt})
        active.remove(node)

    for gene in genes:
        if gene not in visited:
            dfs(gene)

    unresolved = set(genes)
    gene_to_level: dict[str, int] = {}
    level_to_genes: dict[int, list[str]] = {}
    current = list(masters)
    if current:
        level_to_genes[0] = sorted(current)
        for gene in current:
            gene_to_level[gene] = 0
            unresolved.discard(gene)

    level = 1
    while unresolved:
        current_set = {gene for gene in unresolved if incoming[gene].issubset(set(gene_to_level))}
        if not current_set:
            break
        level_to_genes[level] = sorted(current_set)
        for gene in current_set:
            gene_to_level[gene] = level
            unresolved.discard(gene)
        level += 1

    cyclic = bool(unresolved) or bool(autoregulated) or bool(cycle_nodes)
    warnings: list[str] = []
    if autoregulated:
        warnings.append(f"autoregulation detected for genes: {sorted(set(autoregulated))}")
    if cycle_nodes:
        warnings.append(f"directed cycle detected involving genes: {sorted(cycle_nodes)}")
    if unresolved:
        warnings.append(f"cycle or unresolved dependency detected for genes: {sorted(unresolved)}")

    return {
        "master_regulators": masters,
        "target_genes": tuple(gene for gene in genes if gene not in set(masters)),
        "gene_to_level": gene_to_level,
        "level_to_genes": {level: tuple(genes_) for level, genes_ in sorted(level_to_genes.items())},
        "cyclic_or_acyclic": "cyclic" if cyclic else "acyclic",
        "autoregulated_genes": tuple(sorted(set(autoregulated))),
        "cycle_genes": tuple(sorted(cycle_nodes)),
        "unresolved_genes": tuple(sorted(unresolved)),
        "warnings": tuple(warnings),
    }


def calibrate_half_response(
    grn: GRN | pd.DataFrame,
    regulator_expression: pd.Series | pd.DataFrame | dict[str, float],
) -> GRN | pd.DataFrame:
    """Set half-response values from regulator mean expression.

    This mirrors the key SERGIO idea that each edge's half-response is derived
    from the corresponding regulator's mean expression. If a DataFrame is
    provided, means are computed across rows so columns should be gene ids.
    Legacy input column ``threshold`` is accepted, but calibrated outputs only
    keep canonical ``half_response``.
    """

    means = _coerce_mean_expression(regulator_expression)
    if isinstance(grn, GRN):
        edges = grn.to_dataframe()
    else:
        edges = validate_grn(grn)

    missing = sorted(set(edges["regulator"]) - set(means.index))
    if missing:
        raise ValueError(f"missing mean expression for regulators: {missing}")

    calibrated = edges.copy()
    calibrated["half_response"] = calibrated["regulator"].map(means).astype(float)
    if isinstance(grn, GRN):
        return GRN.from_dataframe(calibrated, genes=grn.genes, master_regulators=grn.master_regulators)
    return calibrated


def estimate_state_mean_expression(
    grn: GRN,
    state_production: pd.DataFrame,
    *,
    explicit_master_regulators: Iterable[str] | None = None,
    target_leak_alpha: float | pd.Series | dict[str, float] = 0.0,
    fallback_half_response: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Estimate state-wise mean expression proxies for half-response calibration.

    For acyclic GRNs, this follows graph levels from masters outward. For cyclic
    GRNs, a fallback expression table is returned and metadata marks the
    approximation.
    """

    state_production = state_production.copy()
    state_production.index = state_production.index.astype(str)
    state_production.columns = state_production.columns.astype(str)
    state_production = state_production.astype(float)

    levels = build_graph_levels(grn, explicit_master_regulators=explicit_master_regulators)
    masters = tuple(levels["master_regulators"])
    state_production = state_production.reindex(columns=list(masters), fill_value=0.0)
    means = pd.DataFrame(0.0, index=state_production.index, columns=list(grn.genes), dtype=float)
    means.loc[:, masters] = state_production.loc[:, masters]

    if np.isscalar(target_leak_alpha):
        leak = pd.Series(float(target_leak_alpha), index=pd.Index(grn.genes, name="gene"), dtype=float)
    else:
        leak = pd.Series(target_leak_alpha, dtype=float).reindex(grn.genes, fill_value=0.0)
    leak = leak.clip(lower=0.0)

    if levels["cyclic_or_acyclic"] == "cyclic":
        fallback = pd.DataFrame(fallback_half_response, index=state_production.index, columns=list(grn.genes), dtype=float)
        fallback.loc[:, masters] = state_production.loc[:, masters]
        return fallback, levels

    for level in sorted(levels["level_to_genes"]):
        if level == 0:
            continue
        for gene in levels["level_to_genes"][level]:
            gene_edges = grn.edges.loc[grn.edges["target"].astype(str) == str(gene)]
            gene_mean = pd.Series(float(leak.get(gene, 0.0)), index=state_production.index, dtype=float)
            for edge in gene_edges.itertuples(index=False):
                reg_values = means.loc[:, str(edge.regulator)].to_numpy(dtype=float)
                half_response = edge.half_response if not pd.isna(edge.half_response) else fallback_half_response
                numerator = np.power(reg_values, edge.hill_coefficient)
                denominator = np.power(half_response, edge.hill_coefficient) + numerator
                act = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0)
                response = 1.0 - act if edge.sign == "repression" else act
                gene_mean = gene_mean + float(edge.K) * response
            means.loc[:, str(gene)] = np.maximum(gene_mean.to_numpy(dtype=float), 0.0)

    return means, levels




def estimate_state_mean_expression_cyclic(
    grn: GRN,
    state_production: pd.DataFrame,
    *,
    explicit_master_regulators: Iterable[str] | None = None,
    target_leak_alpha: float | pd.Series | dict[str, float] = 0.0,
    fallback_half_response: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Estimate state-wise activity via fixed-point iteration for cyclic GRNs."""

    state_production = state_production.copy()
    state_production.index = state_production.index.astype(str)
    state_production.columns = state_production.columns.astype(str)
    state_production = state_production.astype(float)

    levels = build_graph_levels(grn, explicit_master_regulators=explicit_master_regulators)
    masters = tuple(levels["master_regulators"])
    state_production = state_production.reindex(columns=list(masters), fill_value=0.0)
    means = pd.DataFrame(fallback_half_response, index=state_production.index, columns=list(grn.genes), dtype=float)
    means.loc[:, masters] = state_production.loc[:, masters]

    if np.isscalar(target_leak_alpha):
        leak = pd.Series(float(target_leak_alpha), index=pd.Index(grn.genes, name="gene"), dtype=float)
    else:
        leak = pd.Series(target_leak_alpha, dtype=float).reindex(grn.genes, fill_value=0.0)
    leak = leak.clip(lower=0.0)

    master_set = set(masters)
    target_to_edges: dict[str, list[object]] = {
        str(gene): list(grn.edges.loc[grn.edges["target"].astype(str) == str(gene)].itertuples(index=False))
        for gene in grn.genes
    }
    status: dict[str, dict[str, object]] = {}
    failed_conditions: list[str] = []
    max_residual = 0.0
    max_iterations = 0

    for state in state_production.index.astype(str):
        current = means.loc[state].astype(float).copy()
        current.loc[list(masters)] = state_production.loc[state, list(masters)].to_numpy(dtype=float)
        converged = False
        residual = float("inf")
        iterations = 0
        for iterations in range(1, max_iter + 1):
            updated = current.copy()
            updated.loc[list(masters)] = state_production.loc[state, list(masters)].to_numpy(dtype=float)
            for gene in grn.genes:
                if gene in master_set:
                    continue
                gene_value = float(leak.get(gene, 0.0))
                for edge in target_to_edges[str(gene)]:
                    reg_value = max(float(current.get(str(edge.regulator), 0.0)), 0.0)
                    half_response = edge.half_response if not pd.isna(edge.half_response) else fallback_half_response
                    numerator = np.power(reg_value, edge.hill_coefficient)
                    denominator = np.power(half_response, edge.hill_coefficient) + numerator
                    act = float(numerator / denominator) if denominator > 0 else 0.0
                    response = 1.0 - act if edge.sign == "repression" else act
                    gene_value += float(edge.K) * response
                updated.loc[str(gene)] = max(gene_value, 0.0)
            residual = float(np.max(np.abs(updated.to_numpy(dtype=float) - current.to_numpy(dtype=float))))
            current = updated
            if residual <= tol:
                converged = True
                break
        means.loc[state] = current.reindex(grn.genes).to_numpy(dtype=float)
        status[state] = {
            "converged": bool(converged),
            "iterations": int(iterations),
            "max_residual": float(residual),
        }
        if not converged:
            failed_conditions.append(str(state))
        max_residual = max(max_residual, float(residual))
        max_iterations = max(max_iterations, int(iterations))

    levels = dict(levels)
    levels["converged"] = len(failed_conditions) == 0
    levels["iterations"] = max_iterations
    levels["max_residual"] = max_residual
    levels["failed_conditions"] = tuple(failed_conditions)
    levels["per_condition_solver_status"] = status
    return means, levels


def _coerce_half_response_calibration(calibration: str) -> str:
    value = str(calibration).strip().lower()
    if value not in HALF_RESPONSE_CALIBRATIONS - {"off"}:
        raise ValueError(
            "half_response calibration must be one of ['auto', 'topology_propagation', 'cyclic']"
        )
    return value


def _half_response_from_values(values: np.ndarray, *, method: str, fallback_half_response: float) -> float:
    regulator_values = np.asarray(values, dtype=float)
    nonzero = regulator_values[regulator_values > 0]
    if method == "median_nonzero":
        value = float(np.nanmedian(nonzero)) if nonzero.size else float(fallback_half_response)
    else:
        value = float(np.nanmean(regulator_values)) if regulator_values.size else float(fallback_half_response)
    if not np.isfinite(value) or value <= 0:
        value = float(fallback_half_response)
    return value


def _resolve_calibration_summary(
    grn_obj: GRN,
    state_production: pd.DataFrame,
    *,
    explicit_master_regulators: Iterable[str] | None,
    calibration: str,
    method: str,
    fallback_half_response: float,
    target_leak_alpha: float | pd.Series | dict[str, float],
    max_iter: int,
    tol: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    level_info = build_graph_levels(grn_obj, explicit_master_regulators=explicit_master_regulators)
    structure = str(level_info["cyclic_or_acyclic"])
    requested = _coerce_half_response_calibration(calibration)
    edges = grn_obj.to_dataframe()
    if requested == "auto":
        if not edges["half_response"].isna().any():
            return pd.DataFrame(), {
                "requested_calibration": "auto",
                "actual_calibration": "provided",
                "calibration_method": "provided",
                "reason": "all_half_response_present",
                "grn_structure": structure,
                "master_regulators": list(level_info["master_regulators"]),
                "gene_levels": dict(level_info["gene_to_level"]),
                "warnings": list(level_info["warnings"]),
                "converged": True,
                "iterations": 0,
                "max_residual": 0.0,
                "failed_conditions": [],
                "failed_genes": [],
                "conditions": [str(state) for state in state_production.index.astype(str)],
                "half_responses_filled_count": 0,
                "half_responses_missing_count_before": int(edges["half_response"].isna().sum()),
                "half_responses_missing_count_after": int(edges["half_response"].isna().sum()),
            }
        actual = "cyclic" if structure == "cyclic" else "topology_propagation"
        reason = "auto_selected_from_grn_structure"
    else:
        actual = requested
        reason = "explicit_request"

    if actual == "topology_propagation":
        if structure == "cyclic":
            raise ValueError("half_response_calibration=\"topology_propagation\" requires an acyclic GRN")
        state_means, meta = estimate_state_mean_expression(
            grn_obj,
            state_production,
            explicit_master_regulators=explicit_master_regulators,
            target_leak_alpha=target_leak_alpha,
            fallback_half_response=fallback_half_response,
        )
        meta = dict(meta)
        meta.setdefault("converged", True)
        meta.setdefault("iterations", 1)
        meta.setdefault("max_residual", 0.0)
        meta.setdefault("failed_conditions", tuple())
    else:
        state_means, meta = estimate_state_mean_expression_cyclic(
            grn_obj,
            state_production,
            explicit_master_regulators=explicit_master_regulators,
            target_leak_alpha=target_leak_alpha,
            fallback_half_response=fallback_half_response,
            max_iter=max_iter,
            tol=tol,
        )
        meta = dict(meta)

    meta["requested_calibration"] = requested
    meta["actual_calibration"] = actual
    meta["calibration_method"] = actual
    meta["reason"] = reason
    meta["grn_structure"] = structure
    meta["conditions"] = [str(state) for state in state_production.index.astype(str)]
    meta["failed_genes"] = []
    return state_means, meta

def calibrate_grn_half_response(
    grn: GRN | pd.DataFrame,
    state_production: pd.DataFrame,
    *,
    explicit_master_regulators: Iterable[str] | None = None,
    calibration: str = "auto",
    method: str = "mean",
    fallback_half_response: float = 1.0,
    target_leak_alpha: float | pd.Series | dict[str, float] = 0.0,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> tuple[GRN | pd.DataFrame, dict[str, object]]:
    """Calibrate half-response values using state-wise regulator activity proxies."""

    if isinstance(grn, GRN):
        grn_obj = grn
        edges = grn.to_dataframe()
    else:
        edges = validate_grn(grn)
        grn_obj = GRN.from_dataframe(edges)

    state_production = state_production.copy()
    state_production.index = state_production.index.astype(str)
    state_production.columns = state_production.columns.astype(str)
    state_production = state_production.astype(float)

    state_means, metadata = _resolve_calibration_summary(
        grn_obj,
        state_production,
        explicit_master_regulators=explicit_master_regulators,
        calibration=calibration,
        method=method,
        fallback_half_response=fallback_half_response,
        target_leak_alpha=target_leak_alpha,
        max_iter=max_iter,
        tol=tol,
    )

    calibrated = edges.copy()
    missing_before = int(calibrated["half_response"].isna().sum())
    filled = 0

    if metadata["actual_calibration"] == "provided":
        if isinstance(grn, GRN):
            return GRN.from_dataframe(calibrated, genes=grn.genes, master_regulators=grn.master_regulators), metadata
        return calibrated, metadata

    overwrite_all = calibration in {"topology_propagation", "cyclic"}
    for idx, edge in calibrated.iterrows():
        if not overwrite_all and pd.notna(edge["half_response"]):
            continue
        regulator_values = state_means[str(edge["regulator"])].to_numpy(dtype=float)
        value = _half_response_from_values(
            regulator_values,
            method=method,
            fallback_half_response=fallback_half_response,
        )
        if pd.isna(calibrated.loc[idx, "half_response"]):
            filled += 1
        calibrated.loc[idx, "half_response"] = value

    metadata["half_responses_filled_count"] = int(filled)
    metadata["half_responses_missing_count_before"] = missing_before
    metadata["half_responses_missing_count_after"] = int(calibrated["half_response"].isna().sum())

    if isinstance(grn, GRN):
        return GRN.from_dataframe(calibrated, genes=grn.genes, master_regulators=grn.master_regulators), metadata
    return calibrated, metadata


def calibrate_grn_thresholds(
    grn: GRN | pd.DataFrame,
    state_production: pd.DataFrame,
    *,
    explicit_master_regulators: Iterable[str] | None = None,
    method: str = "mean",
    fallback_half_response: float = 1.0,
    target_leak_alpha: float | pd.Series | dict[str, float] = 0.0,
) -> tuple[GRN | pd.DataFrame, dict[str, object]]:
    """Legacy alias for half-response calibration."""

    warnings.warn(
        "calibrate_grn_thresholds() is deprecated; use calibrate_grn_half_response() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return calibrate_grn_half_response(
        grn,
        state_production,
        explicit_master_regulators=explicit_master_regulators,
        method=method,
        fallback_half_response=fallback_half_response,
        target_leak_alpha=target_leak_alpha,
    )


@dataclass(frozen=True)
class GRN:
    """一个已经校验过的 GRN 对象。

    ``edges`` 保存标准化后的边表；``genes`` 保存全体基因的固定顺序。
    这个顺序非常重要，因为模拟中的矩阵都是按照 ``genes`` 的顺序排列。
    """

    edges: pd.DataFrame
    genes: tuple[str, ...]
    master_regulators: tuple[str, ...] | None = None

    @classmethod
    def from_dataframe(
        cls,
        edges: pd.DataFrame,
        genes: Iterable[str] | None = None,
        master_regulators: Iterable[str] | None = None,
        config: GRNConfig | None = None,
    ) -> "GRN":
        normalized = validate_grn(edges, config=config)
        if genes is None:
            gene_set = set(normalized["regulator"]).union(normalized["target"])
            ordered_genes = tuple(sorted(gene_set))
        else:
            ordered_genes = tuple(str(gene) for gene in genes)
            unknown = set(normalized["regulator"]).union(normalized["target"]) - set(ordered_genes)
            if unknown:
                raise ValueError(f"GRN contains genes absent from genes list: {sorted(unknown)}")
        normalized_masters: tuple[str, ...] | None = None
        if master_regulators is not None:
            normalized_masters = tuple(str(gene) for gene in master_regulators)
            missing_masters = sorted(set(normalized_masters) - set(ordered_genes))
            if missing_masters:
                raise ValueError(f"master regulators absent from genes list: {missing_masters}")
        return cls(edges=normalized, genes=ordered_genes, master_regulators=normalized_masters)

    def to_dataframe(self) -> pd.DataFrame:
        """返回标准化边表的副本，避免外部代码直接修改 GRN 内部状态。"""

        return self.edges.copy()
