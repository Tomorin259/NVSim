"""NVSim 的 GRN（gene regulatory network）表示与校验。

本模块只负责“网络是什么”，不负责真正计算动力学。
NVSim 当前把 SERGIO-style GRN 参数标准化成：

- regulator
- target
- sign
- K
- half_response
- hill_coefficient

为了兼容旧接口，``weight`` 作为 ``K`` 的别名保留，``threshold`` 作为
``half_response`` 的别名保留。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import GRNConfig

REQUIRED_COLUMNS = ("regulator", "target", "sign")
OPTIONAL_COLUMNS = ("K", "weight", "hill_coefficient", "half_response", "threshold")
RETURN_COLUMNS = ("regulator", "target", "sign", "K", "weight", "half_response", "threshold", "hill_coefficient")
VALID_SIGNS = {"activation", "repression"}
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

    返回值仍然是 DataFrame，但列顺序固定。``threshold`` 是旧接口兼容别名，
    会和 ``half_response`` 保持同步。若 ``half_response`` 缺失，则保留为
    ``NaN``，留给后续校准流程填充。
    """

    config = config or GRNConfig()
    missing = [col for col in REQUIRED_COLUMNS if col not in edges.columns]
    if missing:
        raise ValueError(f"GRN is missing required columns: {missing}")
    if "K" not in edges.columns and "weight" not in edges.columns:
        raise ValueError("GRN must contain either 'K' or 'weight'")

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
    if "weight" not in normalized.columns:
        normalized["weight"] = normalized["K"]

    K_values = normalized["K"].to_numpy(dtype=float)
    weight_values = normalized["weight"].to_numpy(dtype=float)
    if not np.isfinite(K_values).all() or not np.isfinite(weight_values).all():
        raise ValueError("GRN K/weight values must be finite")
    if (K_values < 0).any() or (weight_values < 0).any():
        raise ValueError("GRN K/weight values must be non-negative; sign controls activation/repression")
    if not np.allclose(K_values, weight_values):
        raise ValueError("GRN columns 'K' and 'weight' must match when both are provided")

    if "hill_coefficient" not in normalized.columns:
        normalized["hill_coefficient"] = config.default_hill_coefficient
    if "half_response" not in normalized.columns and "threshold" in normalized.columns:
        normalized["half_response"] = normalized["threshold"]
    if "threshold" not in normalized.columns and "half_response" in normalized.columns:
        normalized["threshold"] = normalized["half_response"]
    if "half_response" not in normalized.columns:
        normalized["half_response"] = np.nan
    if "threshold" not in normalized.columns:
        normalized["threshold"] = normalized["half_response"]

    for col in OPTIONAL_COLUMNS:
        if col not in normalized.columns:
            continue
        normalized[col] = pd.to_numeric(normalized[col], errors="raise")
        values = normalized[col].to_numpy(dtype=float)
        if col in {"K", "weight"}:
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
    ``threshold`` is kept synchronized as a backward-compatible alias.
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
    calibrated["threshold"] = calibrated["half_response"]
    calibrated["K"] = calibrated["weight"]

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


def calibrate_grn_thresholds(
    grn: GRN | pd.DataFrame,
    state_production: pd.DataFrame,
    *,
    explicit_master_regulators: Iterable[str] | None = None,
    method: str = "mean",
    fallback_half_response: float = 1.0,
    target_leak_alpha: float | pd.Series | dict[str, float] = 0.0,
) -> tuple[GRN | pd.DataFrame, dict[str, object]]:
    """Calibrate half-response values using state/bin-wise expression proxies."""

    if isinstance(grn, GRN):
        grn_obj = grn
        edges = grn.to_dataframe()
    else:
        edges = validate_grn(grn)
        grn_obj = GRN.from_dataframe(edges)

    state_means, level_info = estimate_state_mean_expression(
        grn_obj,
        state_production,
        explicit_master_regulators=explicit_master_regulators,
        target_leak_alpha=target_leak_alpha,
        fallback_half_response=fallback_half_response,
    )
    calibrated = edges.copy()
    filled = 0
    calibration_method = "fallback_for_cyclic_grn" if level_info["cyclic_or_acyclic"] == "cyclic" else "levelwise_state_mean"
    for idx, edge in calibrated.iterrows():
        if pd.notna(edge["half_response"]):
            continue
        if level_info["cyclic_or_acyclic"] == "cyclic":
            value = float(fallback_half_response)
        else:
            regulator_values = state_means[str(edge["regulator"])].to_numpy(dtype=float)
            nonzero = regulator_values[regulator_values > 0]
            if method == "median_nonzero":
                value = float(np.nanmedian(nonzero)) if nonzero.size else float(fallback_half_response)
            else:
                value = float(np.nanmean(regulator_values)) if regulator_values.size else float(fallback_half_response)
        if not np.isfinite(value) or value <= 0:
            value = float(fallback_half_response)
        calibrated.loc[idx, "half_response"] = value
        calibrated.loc[idx, "threshold"] = value
        filled += 1

    metadata = {
        "calibration_method": calibration_method if len(calibrated) else "none",
        "master_regulators": level_info["master_regulators"],
        "gene_levels": level_info["gene_to_level"],
        "cyclic_or_acyclic": level_info["cyclic_or_acyclic"],
        "thresholds_filled_count": int(filled),
        "warnings": level_info["warnings"],
    }
    if isinstance(grn, GRN):
        return GRN.from_dataframe(calibrated, genes=grn.genes, master_regulators=grn.master_regulators), metadata
    return calibrated, metadata


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
