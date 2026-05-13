"""NVSim MVP 的核心 ODE 模拟器。

本模块把完整建模链串起来：GRN -> alpha(t) -> unspliced u(t) ->
spliced s(t) -> true velocity。核心方程是：

    du/dt = alpha(t) - beta * u
    ds/dt = beta * u - gamma * s

其中 true_velocity 保存 ds/dt，true_velocity_u 保存 du/dt。
所有 graph segment 都复用同一个 ODE 积分器。
"""

from __future__ import annotations

from typing import Callable, Mapping
import warnings

import numpy as np
import pandas as pd

from .grn import GRN, build_graph_levels, calibrate_grn_half_response, estimate_state_mean_expression
from .modes import StateGraph, DifferentiationGraph, coerce_graph
from .noise import _resolve_capture_model_name, generate_observed_counts
from .output import make_result_dict
from .production import AlphaProgram, StateProductionProfile, coerce_programs, constant
from .regulation import compute_alpha


SIMULATION_MODES = {"sergio_differentiation"}
INITIALIZATION_POLICIES = {"parent_steady_state", "parent_terminal"}
SAMPLING_POLICIES = {"uniform_snapshot", "state_transient"}


def _gene_index(genes: list[str] | tuple[str, ...]) -> pd.Index:
    if not genes:
        raise ValueError("genes must not be empty")
    return pd.Index([str(gene) for gene in genes], name="gene")


def validate_positive_vector(values: object, genes: list[str] | tuple[str, ...], name: str) -> pd.Series:
    """Validate a positive gene-indexed vector such as beta or gamma."""

    index = _gene_index(genes)
    if isinstance(values, pd.Series):
        series = values.astype(float).reindex(index)
        if series.isna().any():
            missing = list(series[series.isna()].index)
            raise ValueError(f"{name} is missing genes: {missing}")
    else:
        arr = np.asarray(values, dtype=float)
        if arr.shape != (len(index),):
            raise ValueError(f"{name} must have shape ({len(index)},)")
        series = pd.Series(arr, index=index, name=name)

    arr = series.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    if (arr <= 0).any():
        raise ValueError(f"{name} must be positive")
    series.name = name
    return series


def create_kinetic_vectors(
    genes: list[str] | tuple[str, ...],
    beta: object | None = None,
    gamma: object | None = None,
    beta_range: tuple[float, float] = (0.4, 1.2),
    gamma_range: tuple[float, float] = (0.2, 0.8),
    seed: int | None = 0,
) -> tuple[pd.Series, pd.Series]:
    """Generate or validate gene-specific beta/gamma vectors."""

    index = _gene_index(genes)
    rng = np.random.default_rng(seed)

    if beta is None:
        lo, hi = beta_range
        if lo <= 0 or hi <= lo:
            raise ValueError("beta_range must be positive and increasing")
        beta_series = pd.Series(rng.uniform(lo, hi, size=len(index)), index=index, name="beta")
    else:
        beta_series = validate_positive_vector(beta, tuple(index), "beta")

    if gamma is None:
        lo, hi = gamma_range
        if lo <= 0 or hi <= lo:
            raise ValueError("gamma_range must be positive and increasing")
        gamma_series = pd.Series(rng.uniform(lo, hi, size=len(index)), index=index, name="gamma")
    else:
        gamma_series = validate_positive_vector(gamma, tuple(index), "gamma")

    return beta_series, gamma_series


def initialize_state(
    genes: list[str] | tuple[str, ...],
    u0: object | None = None,
    s0: object | None = None,
) -> tuple[pd.Series, pd.Series]:
    index = _gene_index(genes)

    def build(values: object | None, name: str) -> pd.Series:
        if values is None:
            return pd.Series(np.zeros(len(index), dtype=float), index=index, name=name)
        if isinstance(values, pd.Series):
            series = values.astype(float).reindex(index)
            if series.isna().any():
                missing = list(series[series.isna()].index)
                raise ValueError(f"{name} is missing genes: {missing}")
        else:
            arr = np.asarray(values, dtype=float)
            if arr.shape != (len(index),):
                raise ValueError(f"{name} must have shape ({len(index)},)")
            series = pd.Series(arr, index=index, name=name)
        arr = series.to_numpy(dtype=float)
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must be finite")
        if (arr < 0).any():
            raise ValueError(f"{name} must be non-negative")
        series.name = name
        return series

    return build(u0, "u0"), build(s0, "s0")

def _infer_master_genes(grn: GRN) -> tuple[str, ...]:
    """Fallback master-regulator inference from network topology."""

    incoming = set(grn.edges["target"])
    return tuple(gene for gene in grn.genes if gene not in incoming)


def _resolve_master_genes(
    grn: GRN,
    master_regulators: list[str] | tuple[str, ...] | pd.Index | None = None,
    production_profile: StateProductionProfile | None = None,
    allow_profile_targets_as_masters: bool = False,
    profile_gene_policy: str = "exact",
) -> tuple[tuple[str, ...], dict[str, object]]:
    explicit_provided = master_regulators is not None
    grn_metadata_provided = grn.master_regulators is not None
    if master_regulators is not None:
        masters = tuple(str(gene) for gene in master_regulators)
        source = "explicit"
    elif grn.master_regulators is not None:
        masters = tuple(str(gene) for gene in grn.master_regulators)
        source = "grn_metadata"
    elif production_profile is not None and profile_gene_policy == "exact":
        masters = tuple(str(gene) for gene in production_profile.genes)
        source = "production_profile"
    else:
        masters = _infer_master_genes(grn)
        source = "topology_inference"
    missing = sorted(set(masters) - set(grn.genes))
    if missing:
        raise ValueError(f"master regulators absent from gene list: {missing}")
    incoming_to_masters = grn.edges.loc[grn.edges["target"].astype(str).isin(masters)].copy()
    metadata = {
        "resolved_master_regulator_source": source,
        "incoming_edges_to_masters_count": int(incoming_to_masters.shape[0]),
        "incoming_edges_to_masters": incoming_to_masters.to_dict(orient="records"),
        "allow_profile_targets_as_masters": bool(allow_profile_targets_as_masters),
    }
    conflicting_genes = sorted(set(masters).intersection(set(grn.edges["target"].astype(str))))
    if source == "production_profile" and conflicting_genes and not explicit_provided and not grn_metadata_provided:
        if not allow_profile_targets_as_masters:
            raise ValueError(
                "Genes appearing in both production_profile and GRN targets: "
                f"{conflicting_genes}. If these genes are treated as master regulators, "
                "their incoming regulatory edges will be ignored. Pass explicit "
                "master_regulators, store master_regulators in the GRN metadata, or set "
                "allow_profile_targets_as_masters=True to accept this override explicitly."
            )
        warnings.warn(
            "production_profile genes are being treated as master regulators even though they "
            f"also have incoming GRN edges: {conflicting_genes}. Incoming regulatory edges to "
            "those genes will be ignored.",
            UserWarning,
            stacklevel=2,
        )
    return masters, metadata


def _resolve_alpha_source_mode(
    alpha_source_mode: str | None,
    *,
    production_profile: StateProductionProfile | None = None,
    production_state: str | None = None,
    parent_state: str | None = None,
    child_state: str | None = None,
    state_args_present: bool = False,
) -> str:
    """Resolve the master-regulator alpha source mode.

    ``continuous_program`` is NVSim's original alpha_m(t)=f_m(t) mode.
    ``state_anchor`` uses SERGIO-style state/bin production anchors.
    """

    state_args_present = state_args_present or any(
        value is not None for value in (production_state, parent_state, child_state)
    )
    allowed = {"continuous_program", "state_anchor"}
    if alpha_source_mode is not None:
        if alpha_source_mode not in allowed:
            raise ValueError("alpha_source_mode must be 'continuous_program' or 'state_anchor'")
        if state_args_present and alpha_source_mode != "state_anchor":
            raise ValueError("state_anchor state arguments require alpha_source_mode='state_anchor'")
        return alpha_source_mode
    if state_args_present:
        return "state_anchor"
    if production_profile is not None:
        return "state_anchor"
    return "continuous_program"


def _serialize_master_programs(programs: Mapping[str, AlphaProgram]) -> dict[str, dict[str, float | str]]:
    """Return a metadata-safe representation of master AlphaProgram objects."""

    return {
        str(gene): {
            "kind": program.kind,
            "start": float(program.start),
            "end": float(program.end),
            "midpoint": float(program.midpoint),
            "steepness": float(program.steepness),
        }
        for gene, program in programs.items()
    }


def _validate_profile_genes(
    production_profile: StateProductionProfile,
    master_genes: tuple[str, ...],
    profile_gene_policy: str,
) -> None:
    if profile_gene_policy == "exact":
        production_profile.validate_master_genes(master_genes)
        return
    if profile_gene_policy != "subset_fill":
        raise ValueError("profile_gene_policy must be 'exact' or 'subset_fill'")
    expected = {str(gene) for gene in master_genes}
    observed = set(production_profile.genes)
    extra = sorted(observed - expected)
    if extra:
        raise ValueError("production profile genes do not match master genes: extra=" + str(extra))


def _complete_source_alpha(
    alpha: pd.Series,
    genes: tuple[str, ...],
    master_genes: tuple[str, ...],
    default_master_alpha: float,
    profile_gene_policy: str,
) -> pd.Series:
    completed = pd.Series(0.0, index=pd.Index(genes, name="gene"), dtype=float)
    completed.update(alpha.astype(float))
    if profile_gene_policy == "subset_fill":
        missing_masters = [gene for gene in master_genes if gene not in alpha.index]
        if missing_masters:
            completed.loc[missing_masters] = float(default_master_alpha)
    return completed


def _source_alpha_from_profile(
    production_profile: StateProductionProfile,
    state: str,
    genes: tuple[str, ...],
    master_genes: tuple[str, ...],
    default_master_alpha: float,
    profile_gene_policy: str,
) -> pd.Series:
    alpha = production_profile.source_alpha(state)
    return _complete_source_alpha(alpha, genes, master_genes, default_master_alpha, profile_gene_policy)


def _state_transition_source_alpha_fn(
    production_profile: StateProductionProfile,
    parent_state: str,
    child_state: str,
    segment_start_time: float,
    segment_time: float,
    genes: tuple[str, ...],
    master_genes: tuple[str, ...],
    default_master_alpha: float,
    profile_gene_policy: str,
    transition_schedule: str,
    transition_midpoint: float,
    transition_steepness: float,
) -> Callable[[float], pd.Series]:
    """Create a state-anchor parent-to-child source-alpha function."""

    def resolve(global_t: float) -> pd.Series:
        if segment_time <= 0:
            fraction = 1.0
        else:
            fraction = float(np.clip((global_t - segment_start_time) / segment_time, 0.0, 1.0))
        alpha = production_profile.source_alpha_transition(
            parent_state,
            child_state,
            fraction,
            schedule=transition_schedule,
            midpoint=transition_midpoint,
            steepness=transition_steepness,
        )
        return _complete_source_alpha(alpha, genes, master_genes, default_master_alpha, profile_gene_policy)

    return resolve


def _coerce_branch_state_mapping(
    branch_states: Mapping[str, str] | tuple[str, str] | list[str],
    branch_labels: tuple[str, str],
    *,
    parameter_name: str,
) -> dict[str, str]:
    if isinstance(branch_states, Mapping):
        missing = [branch for branch in branch_labels if branch not in branch_states]
        if missing:
            raise ValueError(f"{parameter_name} missing branches: {missing}")
        return {branch: str(branch_states[branch]) for branch in branch_labels}
    if len(branch_states) != len(branch_labels):
        raise ValueError(f"{parameter_name} must contain exactly two states for branch_0 and branch_1")
    return {branch: str(state) for branch, state in zip(branch_labels, branch_states)}


def _resolve_branch_child_states(
    branch_child_states: Mapping[str, str] | tuple[str, str] | list[str] | None,
    branch_labels: tuple[str, str],
) -> dict[str, str] | None:
    if branch_child_states is not None:
        return _coerce_branch_state_mapping(
            branch_child_states,
            branch_labels,
            parameter_name="branch_child_states",
        )
    return None


def _warn_legacy_bifurcation_state_args(
    *,
    trunk_production_state: str | None,
    branch_production_states: Mapping[str, str] | None,
    interpolate_production: bool,
) -> None:
    legacy_args: list[str] = []
    if trunk_production_state is not None:
        legacy_args.append("trunk_production_state")
    if branch_production_states is not None:
        legacy_args.append("branch_production_states")
    if interpolate_production:
        legacy_args.append("interpolate_production")
    if legacy_args:
        warnings.warn(
            "Legacy bifurcation production-profile arguments "
            + ", ".join(legacy_args)
            + " are deprecated; prefer trunk_state + branch_child_states + transition_schedule.",
            DeprecationWarning,
            stacklevel=3,
        )


def _alpha_from_state(
    u: np.ndarray,
    s: np.ndarray,
    t: float,
    time_end: float,
    grn: GRN,
    master_genes: tuple[str, ...],
    master_programs: Mapping[str, AlphaProgram],
    default_master_alpha: float,
    alpha_max: float | None,
    target_leak_alpha: pd.Series | dict[str, float] | float = 0.0,
    source_alpha: pd.Series | None = None,
    source_alpha_fn: Callable[[float], pd.Series] | None = None,
    regulator_activity: str = "spliced",
    return_edge_contributions: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    # 根据当前 spliced 状态 s(t) 重新计算 alpha(t)。
    # master 的 alpha 优先来自 source_alpha；否则来自时间程序。
    # target/intermediate 的 alpha 来自 GRN。
    genes = grn.genes
    normalized_t = 0.0 if time_end == 0 else float(np.clip(t / time_end, 0.0, 1.0))
    source = pd.Series(0.0, index=pd.Index(genes, name="gene"), dtype=float)
    if source_alpha_fn is not None:
        source.update(source_alpha_fn(t).reindex(genes, fill_value=0.0))
    elif source_alpha is not None:
        source.update(source_alpha.reindex(genes, fill_value=0.0))
    else:
        for gene in master_genes:
            program = master_programs.get(gene, constant(default_master_alpha))
            source.loc[gene] = program.value(normalized_t)
    if regulator_activity == "spliced":
        activity_values = np.maximum(s, 0.0)
    elif regulator_activity == "unspliced":
        activity_values = np.maximum(u, 0.0)
    elif regulator_activity == "total":
        activity_values = np.maximum(u, 0.0) + np.maximum(s, 0.0)
    else:
        raise ValueError("regulator_activity must be 'spliced', 'unspliced', or 'total'")
    regulator_values = pd.Series(activity_values, index=pd.Index(genes, name="gene"), dtype=float)
    computed = compute_alpha(
        regulator_values,
        grn,
        source_alpha=source,
        target_leak_alpha=target_leak_alpha,
        master_regulators=master_genes,
        alpha_min=0.0,
        alpha_max=alpha_max,
        return_edge_contributions=return_edge_contributions,
    )
    if return_edge_contributions:
        alpha, edge_contributions = computed
        return alpha.reindex(genes).to_numpy(dtype=float), edge_contributions.to_numpy(dtype=float)
    alpha = computed
    return alpha.reindex(genes).to_numpy(dtype=float)


def _derivative(
    y: np.ndarray,
    t: float,
    time_end: float,
    grn: GRN,
    master_genes: tuple[str, ...],
    beta: np.ndarray,
    gamma: np.ndarray,
    master_programs: Mapping[str, AlphaProgram],
    default_master_alpha: float,
    alpha_max: float | None,
    target_leak_alpha: pd.Series | dict[str, float] | float = 0.0,
    source_alpha: pd.Series | None = None,
    source_alpha_fn: Callable[[float], pd.Series] | None = None,
    regulator_activity: str = "spliced",
) -> tuple[np.ndarray, np.ndarray]:
    # y 是拼接状态向量：[u_1...u_G, s_1...s_G]。
    n_genes = len(beta)
    u = np.maximum(y[:n_genes], 0.0)
    s = np.maximum(y[n_genes:], 0.0)
    alpha = _alpha_from_state(
        u,
        s,
        t,
        time_end,
        grn,
        master_genes,
        master_programs,
        default_master_alpha,
        alpha_max,
        target_leak_alpha,
        source_alpha,
        source_alpha_fn,
        regulator_activity,
    )
    # RNA velocity 方程：du/dt -> true_velocity_u；ds/dt -> true_velocity。
    du = alpha - beta * u
    ds = beta * u - gamma * s
    return np.concatenate([du, ds]), alpha


def _rk4_step(
    y: np.ndarray,
    t: float,
    dt: float,
    time_end: float,
    grn: GRN,
    master_genes: tuple[str, ...],
    beta: np.ndarray,
    gamma: np.ndarray,
    master_programs: Mapping[str, AlphaProgram],
    default_master_alpha: float,
    alpha_max: float | None,
    target_leak_alpha: pd.Series | dict[str, float] | float = 0.0,
    source_alpha: pd.Series | None = None,
    source_alpha_fn: Callable[[float], pd.Series] | None = None,
    regulator_activity: str = "spliced",
) -> np.ndarray:
    """执行一步 RK4；每个中间状态都会重新通过 GRN 计算 alpha。"""

    k1, _ = _derivative(
        y,
        t,
        time_end,
        grn,
        master_genes,
        beta,
        gamma,
        master_programs,
        default_master_alpha,
        alpha_max,
        target_leak_alpha,
        source_alpha,
        source_alpha_fn,
        regulator_activity,
    )
    k2, _ = _derivative(
        y + 0.5 * dt * k1,
        t + 0.5 * dt,
        time_end,
        grn,
        master_genes,
        beta,
        gamma,
        master_programs,
        default_master_alpha,
        alpha_max,
        target_leak_alpha,
        source_alpha,
        source_alpha_fn,
        regulator_activity,
    )
    k3, _ = _derivative(
        y + 0.5 * dt * k2,
        t + 0.5 * dt,
        time_end,
        grn,
        master_genes,
        beta,
        gamma,
        master_programs,
        default_master_alpha,
        alpha_max,
        target_leak_alpha,
        source_alpha,
        source_alpha_fn,
        regulator_activity,
    )
    k4, _ = _derivative(
        y + dt * k3,
        t + dt,
        time_end,
        grn,
        master_genes,
        beta,
        gamma,
        master_programs,
        default_master_alpha,
        alpha_max,
        target_leak_alpha,
        source_alpha,
        source_alpha_fn,
        regulator_activity,
    )
    y_next = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.maximum(y_next, 0.0)


def _time_grid(time_end: float, dt: float) -> tuple[np.ndarray, float]:
    if time_end <= 0:
        raise ValueError("time_end must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")
    n_steps = int(np.ceil(time_end / dt)) + 1
    times = np.linspace(0.0, time_end, n_steps)
    actual_dt = times[1] - times[0] if n_steps > 1 else time_end
    return times, actual_dt


def _simulate_segment(
    *,
    grn: GRN,
    master_genes: tuple[str, ...],
    beta: np.ndarray,
    gamma: np.ndarray,
    u0: np.ndarray,
    s0: np.ndarray,
    time_end: float,
    dt: float,
    master_programs: Mapping[str, AlphaProgram],
    default_master_alpha: float,
    alpha_max: float | None,
    target_leak_alpha: pd.Series | dict[str, float] | float = 0.0,
    time_offset: float = 0.0,
    program_time_end: float | None = None,
    source_alpha: pd.Series | None = None,
    source_alpha_fn: Callable[[float], pd.Series] | None = None,
    regulator_activity: str = "spliced",
    return_edge_contributions: bool = False,
) -> dict[str, np.ndarray]:
    """Simulate one linear segment and return timepoints x genes arrays.

    ``local_time`` starts at zero for the segment. ``pseudotime`` is
    ``time_offset + local_time`` and is used to evaluate master-regulator
    programs against ``program_time_end``. This lets trunk and branch segments
    share one global program clock while preserving segment-local metadata.
    """

    genes = grn.genes
    n_genes = len(genes)
    if u0.shape != (n_genes,) or s0.shape != (n_genes,):
        raise ValueError("u0 and s0 must be vectors with one value per gene")

    local_time, actual_dt = _time_grid(time_end, dt)
    global_time = time_offset + local_time
    alpha_time_end = program_time_end if program_time_end is not None else time_offset + time_end

    # 内部 time-course 是 timepoints x genes；采样后输出才是 cells x genes。
    u = np.zeros((len(local_time), n_genes), dtype=float)
    s = np.zeros((len(local_time), n_genes), dtype=float)
    alpha = np.zeros((len(local_time), n_genes), dtype=float)
    velocity = np.zeros((len(local_time), n_genes), dtype=float)
    true_velocity_u = np.zeros((len(local_time), n_genes), dtype=float)
    edge_contributions = (
        np.zeros((len(local_time), grn.edges.shape[0]), dtype=float) if return_edge_contributions else None
    )

    u[0] = np.maximum(u0, 0.0)
    s[0] = np.maximum(s0, 0.0)
    alpha0 = _alpha_from_state(
        u[0],
        s[0],
        global_time[0],
        alpha_time_end,
        grn,
        master_genes,
        master_programs,
        default_master_alpha,
        alpha_max,
        target_leak_alpha,
        source_alpha,
        source_alpha_fn,
        regulator_activity,
        return_edge_contributions=return_edge_contributions,
    )
    if return_edge_contributions:
        alpha[0], edge_contributions[0] = alpha0
    else:
        alpha[0] = alpha0
    # true_velocity 是 spliced velocity ds/dt；true_velocity_u 是 unspliced velocity du/dt。
    velocity[0] = beta * u[0] - gamma * s[0]
    true_velocity_u[0] = alpha[0] - beta * u[0]

    y = np.concatenate([u[0], s[0]])
    for step in range(1, len(local_time)):
        y = _rk4_step(
            y,
            global_time[step - 1],
            actual_dt,
            alpha_time_end,
            grn,
            master_genes,
            beta,
            gamma,
            master_programs,
            default_master_alpha,
            alpha_max,
            target_leak_alpha,
            source_alpha,
            source_alpha_fn,
            regulator_activity,
        )
        u[step] = y[:n_genes]
        s[step] = y[n_genes:]
        alpha_step = _alpha_from_state(
            u[step],
            s[step],
            global_time[step],
            alpha_time_end,
            grn,
            master_genes,
            master_programs,
            default_master_alpha,
            alpha_max,
            target_leak_alpha,
            source_alpha,
            source_alpha_fn,
            regulator_activity,
            return_edge_contributions=return_edge_contributions,
        )
        if return_edge_contributions:
            alpha[step], edge_contributions[step] = alpha_step
        else:
            alpha[step] = alpha_step
        velocity[step] = beta * u[step] - gamma * s[step]
        true_velocity_u[step] = alpha[step] - beta * u[step]

    segment = {
        "local_time": local_time,
        "pseudotime": global_time,
        "actual_dt": np.asarray(actual_dt),
        "u": u,
        "s": s,
        "alpha": alpha,
        "velocity": velocity,
        "true_velocity_u": true_velocity_u,
    }
    if edge_contributions is not None:
        segment["edge_contributions"] = edge_contributions
    return segment


def _sample_snapshots(
    rng: np.random.Generator,
    n_cells: int,
    available_indices: np.ndarray,
    allow_snapshot_replacement: bool = False,
) -> tuple[np.ndarray, dict[str, int | bool]]:
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")
    indices = np.asarray(available_indices, dtype=int)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("available_indices must be a non-empty 1D array")
    replace = n_cells > indices.size
    if replace and not allow_snapshot_replacement:
        raise ValueError(
            "n_cells exceeds available timepoints for snapshot sampling. Increase available timepoints, "
            "reduce dt, reduce n_cells, or set allow_snapshot_replacement=True."
        )
    sampled = rng.choice(indices, size=n_cells, replace=replace)
    sampled = np.sort(sampled)
    unique_count = int(np.unique(sampled).size)
    return sampled, {
        "sampling_replace": bool(replace),
        "n_unique_timepoints_sampled": unique_count,
        "n_duplicate_snapshot_cells": int(n_cells - unique_count),
    }


def _coerce_state_scalar_mapping(
    values: int | float | Mapping[str, int | float],
    states: tuple[str, ...],
    *,
    parameter_name: str,
    cast: Callable[[int | float], int | float],
    positive: bool = True,
) -> dict[str, int | float]:
    if isinstance(values, Mapping):
        missing = [state for state in states if state not in values]
        if missing:
            raise ValueError(f"{parameter_name} missing states: {missing}")
        resolved = {str(state): cast(values[state]) for state in states}
    else:
        resolved = {str(state): cast(values) for state in states}
    if positive:
        invalid = [state for state, value in resolved.items() if value <= 0]
        if invalid:
            raise ValueError(f"{parameter_name} must be positive for all states; invalid states: {invalid}")
    return resolved


def _resolve_mode_defaults(
    *,
    simulator: str | None,
    simulation_mode: str | None,
    graph: StateGraph | pd.DataFrame | dict[str, object] | None,
    production_profile: StateProductionProfile | None,
    alpha_source_mode: str | None,
    initialization_policy: str | None,
    sampling_policy: str | None,
    transition_schedule: str | None,
    regulator_activity: str | None,
    auto_calibrate_half_response: bool | str | None,
) -> dict[str, object]:
    resolved_graph = coerce_graph(graph)
    resolved_mode = None if simulation_mode is None else str(simulation_mode).strip().lower()
    if resolved_mode is not None and resolved_mode not in SIMULATION_MODES:
        raise ValueError(f"simulation_mode must be one of {sorted(SIMULATION_MODES)}")

    resolved_simulator = None if simulator is None else str(simulator).strip().lower()
    resolved_alpha_source_mode = alpha_source_mode
    resolved_initialization_policy = initialization_policy
    resolved_sampling_policy = sampling_policy
    resolved_transition_schedule = transition_schedule
    resolved_regulator_activity = regulator_activity
    resolved_auto_calibration = auto_calibrate_half_response

    if resolved_mode == "sergio_differentiation":
        if production_profile is None:
            raise ValueError("simulation_mode=\"sergio_differentiation\" requires production_profile")
        if resolved_graph is None:
            raise ValueError("simulation_mode=\"sergio_differentiation\" requires graph")
        if resolved_simulator not in (None, "graph"):
            raise ValueError(
                "simulation_mode=\"sergio_differentiation\" is incompatible with simulator="
                f"{resolved_simulator!r}; use simulator=\"graph\" or omit it"
            )
        if resolved_alpha_source_mode not in (None, "state_anchor"):
            raise ValueError("simulation_mode=\"sergio_differentiation\" requires alpha_source_mode=\"state_anchor\"")
        resolved_simulator = "graph"
        resolved_alpha_source_mode = "state_anchor"
        if resolved_initialization_policy is None:
            resolved_initialization_policy = "parent_steady_state"
        if resolved_sampling_policy is None:
            resolved_sampling_policy = "state_transient"
        if resolved_transition_schedule is None:
            resolved_transition_schedule = "step"
        if resolved_regulator_activity is None:
            resolved_regulator_activity = "unspliced"
        if resolved_auto_calibration is None:
            resolved_auto_calibration = "if_missing"

    if resolved_graph is None:
        raise ValueError("graph must be provided; graph is now the only supported simulator topology")
    if resolved_simulator is None:
        resolved_simulator = "graph"
    if resolved_simulator != "graph":
        raise ValueError("simulator must be \"graph\"")
    if resolved_initialization_policy is None:
        resolved_initialization_policy = "parent_terminal"
    if resolved_sampling_policy is None:
        resolved_sampling_policy = "state_transient"
    if resolved_transition_schedule is None:
        resolved_transition_schedule = "sigmoid"
    if resolved_regulator_activity is None:
        resolved_regulator_activity = "spliced"
    if resolved_auto_calibration is None:
        resolved_auto_calibration = False

    if resolved_initialization_policy not in INITIALIZATION_POLICIES:
        raise ValueError(f"initialization_policy must be one of {sorted(INITIALIZATION_POLICIES)}")
    if resolved_sampling_policy not in SAMPLING_POLICIES:
        raise ValueError(f"sampling_policy must be one of {sorted(SAMPLING_POLICIES)}")

    return {
        "simulation_mode": resolved_mode,
        "simulator": resolved_simulator,
        "graph": resolved_graph,
        "alpha_source_mode": resolved_alpha_source_mode,
        "initialization_policy": resolved_initialization_policy,
        "sampling_policy": resolved_sampling_policy,
        "transition_schedule": resolved_transition_schedule,
        "regulator_activity": resolved_regulator_activity,
        "auto_calibrate_half_response": resolved_auto_calibration,
    }

def _steady_state_table_from_source_rates(
    grn: GRN,
    source_rates: pd.DataFrame,
    *,
    master_genes: tuple[str, ...],
    beta: pd.Series,
    gamma: pd.Series,
    target_leak_alpha: pd.Series | dict[str, float] | float = 0.0,
) -> dict[str, dict[str, pd.Series]]:
    state_alpha, _ = estimate_state_mean_expression(
        grn,
        source_rates,
        explicit_master_regulators=master_genes,
        target_leak_alpha=target_leak_alpha,
    )
    state_alpha = state_alpha.reindex(columns=list(grn.genes), fill_value=0.0).astype(float)

    steady_states: dict[str, dict[str, pd.Series]] = {}
    beta_series = beta.reindex(grn.genes).astype(float)
    gamma_series = gamma.reindex(grn.genes).astype(float)
    beta_arr = beta_series.to_numpy(dtype=float)
    gamma_arr = gamma_series.to_numpy(dtype=float)
    for state in source_rates.index.astype(str):
        alpha = state_alpha.loc[str(state)].astype(float)
        u = np.divide(
            alpha.to_numpy(dtype=float),
            beta_arr,
            out=np.zeros(len(grn.genes), dtype=float),
            where=beta_arr > 0,
        )
        s = np.divide(
            beta_arr * u,
            gamma_arr,
            out=np.zeros(len(grn.genes), dtype=float),
            where=gamma_arr > 0,
        )
        steady_states[str(state)] = {
            "alpha": alpha.reindex(grn.genes),
            "u": pd.Series(u, index=pd.Index(grn.genes, name="gene"), dtype=float),
            "s": pd.Series(s, index=pd.Index(grn.genes, name="gene"), dtype=float),
        }
    return steady_states


def _source_rates_from_program_starts(
    *,
    states: tuple[str, ...],
    state_offsets: Mapping[str, float],
    total_program_time: float,
    master_genes: tuple[str, ...],
    base_programs: Mapping[str, AlphaProgram],
    state_master_programs: Mapping[str, Mapping[str, AlphaProgram | float]] | None,
) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    total = max(float(total_program_time), 1e-8)
    for state in states:
        merged = _merge_programs(base_programs, None if state_master_programs is None else state_master_programs.get(state))
        normalized_t = float(np.clip(float(state_offsets[state]) / total, 0.0, 1.0))
        rows[str(state)] = {
            gene: float(merged.get(gene, constant(0.0)).value(normalized_t))
            for gene in master_genes
        }
    return pd.DataFrame.from_dict(rows, orient="index").reindex(index=list(states), columns=list(master_genes), fill_value=0.0)


def _graph_total_program_time(
    states: tuple[str, ...],
    state_offsets: Mapping[str, float],
    state_durations: Mapping[str, float],
    root_states: tuple[str, ...],
    root_time: float,
) -> float:
    maxima = []
    for state in states:
        duration = float(root_time) if state in root_states else float(state_durations[state])
        maxima.append(float(state_offsets[state]) + duration)
    return max(maxima) if maxima else float(root_time)

def _prepare_common_inputs(
    grn: GRN,
    beta: object | None,
    gamma: object | None,
    u0: object | None,
    s0: object | None,
    master_programs: Mapping[str, AlphaProgram | float] | None,
    default_master_alpha: float,
    seed: int,
):
    if default_master_alpha < 0:
        raise ValueError("default_master_alpha must be non-negative")
    genes = grn.genes
    beta_series, gamma_series = create_kinetic_vectors(genes, beta=beta, gamma=gamma, seed=seed)
    u0_series, s0_series = initialize_state(genes, u0=u0, s0=s0)
    return (
        beta_series,
        gamma_series,
        u0_series,
        s0_series,
        beta_series.to_numpy(dtype=float),
        gamma_series.to_numpy(dtype=float),
        u0_series.to_numpy(dtype=float),
        s0_series.to_numpy(dtype=float),
        coerce_programs(master_programs),
    )


def _grn_calibration_summary(
    grn: GRN,
    master_genes: tuple[str, ...],
    grn_calibration: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if grn_calibration is not None:
        return dict(grn_calibration)

    level_info = build_graph_levels(grn, explicit_master_regulators=master_genes)
    return {
        "calibration_method": "not_recorded",
        "master_regulators": list(master_genes),
        "gene_levels": dict(level_info["gene_to_level"]),
        "cyclic_or_acyclic": level_info["cyclic_or_acyclic"],
        "half_responses_filled_count": 0,
        "half_responses_missing_count": int(grn.edges["half_response"].isna().sum()),
        "warnings": list(level_info["warnings"]),
    }


def _maybe_calibrate_grn(
    grn: GRN,
    *,
    master_genes: tuple[str, ...],
    production_profile: StateProductionProfile | None,
    auto_calibrate_half_response: bool | str,
    grn_calibration: Mapping[str, object] | None,
    target_leak_alpha: pd.Series | dict[str, float] | float = 0.0,
) -> tuple[GRN, Mapping[str, object] | None]:
    if auto_calibrate_half_response not in (False, True, "if_missing"):
        raise ValueError("auto_calibrate_half_response must be False, True, or 'if_missing'")
    if auto_calibrate_half_response is False:
        return grn, grn_calibration
    if production_profile is None:
        raise ValueError("production_profile must be provided when auto_calibrate_half_response is enabled")

    missing_half_response = bool(grn.edges["half_response"].isna().any())
    if auto_calibrate_half_response == "if_missing" and not missing_half_response:
        return grn, grn_calibration

    calibrated_grn, calibration = calibrate_grn_half_response(
        grn,
        production_profile.rates,
        explicit_master_regulators=master_genes,
        target_leak_alpha=target_leak_alpha,
    )
    merged = dict(grn_calibration or {})
    merged.update(calibration)
    merged["auto_calibrate_half_response"] = auto_calibrate_half_response
    return calibrated_grn, merged


def _gene_metadata(
    grn: GRN,
    master_genes: tuple[str, ...],
    *,
    beta: pd.Series,
    gamma: pd.Series,
) -> pd.DataFrame:
    var = pd.DataFrame(index=pd.Index(grn.genes, name="gene"))
    master_set = set(master_genes)
    var["gene_role"] = ["master_regulator" if gene in master_set else "target" for gene in grn.genes]
    var["gene_class"] = "unassigned"
    level_info = build_graph_levels(grn, explicit_master_regulators=master_genes)
    gene_level = pd.Series(pd.NA, index=var.index, dtype="Int64")
    for gene, level in level_info["gene_to_level"].items():
        gene_level.loc[gene] = int(level)
    var["gene_level"] = gene_level
    var["true_beta"] = beta.reindex(var.index).to_numpy(dtype=float)
    var["true_gamma"] = gamma.reindex(var.index).to_numpy(dtype=float)
    return var


def _observed_from_true(
    true_u: np.ndarray,
    true_s: np.ndarray,
    *,
    seed: int,
    noise_seed: int | None,
    capture_rate: float | None,
    poisson_observed: bool,
    dropout_rate: float,
    capture_model: str | None = None,
) -> dict[str, np.ndarray]:
    return generate_observed_counts(
        true_u,
        true_s,
        seed=seed + 2 if noise_seed is None else noise_seed,
        capture_rate=capture_rate,
        poisson=poisson_observed,
        dropout_rate=dropout_rate,
        capture_model=capture_model,
    )


def _merge_programs(
    base: Mapping[str, AlphaProgram],
    overrides: Mapping[str, AlphaProgram | float] | None,
) -> dict[str, AlphaProgram]:
    merged = dict(base)
    if overrides:
        merged.update(coerce_programs(overrides))
    return merged

def _simulate_graph_impl(
    grn: GRN,
    n_cells_per_state: int | Mapping[str, int] = 60,
    root_time: float = 2.0,
    state_time: float | Mapping[str, float] = 2.0,
    dt: float = 0.01,
    beta: object | None = None,
    gamma: object | None = None,
    u0: object | None = None,
    s0: object | None = None,
    master_regulators: list[str] | tuple[str, ...] | pd.Index | None = None,
    graph: StateGraph | pd.DataFrame | dict[str, object] | None = None,
    production_profile: StateProductionProfile | None = None,
    master_programs: Mapping[str, AlphaProgram | float] | None = None,
    state_master_programs: Mapping[str, Mapping[str, AlphaProgram | float]] | None = None,
    default_master_alpha: float = 0.5,
    target_leak_alpha: pd.Series | dict[str, float] | float = 0.0,
    alpha_max: float | None = None,
    seed: int = 0,
    noise_seed: int | None = None,
    capture_rate: float | None = None,
    poisson_observed: bool = True,
    dropout_rate: float = 0.0,
    capture_model: str | None = None,
    regulator_activity: str = "spliced",
    auto_calibrate_half_response: bool | str = False,
    grn_calibration: Mapping[str, object] | None = None,
    return_edge_contributions: bool = False,
    allow_profile_targets_as_masters: bool = False,
    allow_snapshot_replacement: bool = False,
    alpha_source_mode: str | None = None,
    initialization_policy: str = "parent_terminal",
    sampling_policy: str = "state_transient",
    transition_schedule: str = "sigmoid",
    transition_midpoint: float = 0.5,
    transition_steepness: float = 10.0,
    profile_gene_policy: str = "exact",
    simulation_mode: str | None = None,
) -> dict:
    resolved_graph = coerce_graph(graph)
    if resolved_graph is None:
        raise ValueError("graph must be provided for simulator graph")
    if initialization_policy not in INITIALIZATION_POLICIES:
        raise ValueError(f"initialization_policy must be one of {sorted(INITIALIZATION_POLICIES)}")
    if sampling_policy not in SAMPLING_POLICIES:
        raise ValueError(f"sampling_policy must be one of {sorted(SAMPLING_POLICIES)}")

    (
        beta_series,
        gamma_series,
        _u0_series,
        _s0_series,
        beta_arr,
        gamma_arr,
        u0_arr,
        s0_arr,
        programs,
    ) = _prepare_common_inputs(grn, beta, gamma, u0, s0, master_programs, default_master_alpha, seed)
    master_genes, master_metadata = _resolve_master_genes(
        grn,
        master_regulators=master_regulators,
        production_profile=production_profile,
        allow_profile_targets_as_masters=allow_profile_targets_as_masters,
        profile_gene_policy=profile_gene_policy,
    )
    states = tuple(str(state) for state in resolved_graph.topological_order())
    state_depths = resolved_graph.state_depths()
    root_states = tuple(str(state) for state in resolved_graph.root_states)
    state_cell_counts = _coerce_state_scalar_mapping(
        n_cells_per_state,
        states,
        parameter_name="n_cells_per_state",
        cast=lambda value: int(value),
    )
    state_durations = _coerce_state_scalar_mapping(
        state_time,
        states,
        parameter_name="state_time",
        cast=lambda value: float(value),
    )

    state_offsets: dict[str, float] = {}
    for state in states:
        parent_state = resolved_graph.parent_of(state)
        if parent_state is None:
            state_offsets[state] = 0.0
        else:
            parent_duration = float(root_time) if parent_state in root_states else float(state_durations[parent_state])
            state_offsets[state] = float(state_offsets[parent_state]) + parent_duration
    total_program_time = _graph_total_program_time(states, state_offsets, state_durations, root_states, root_time)

    resolved_alpha_source_mode = _resolve_alpha_source_mode(
        alpha_source_mode,
        production_profile=production_profile,
    )
    if resolved_alpha_source_mode == "state_anchor":
        if production_profile is None:
            raise ValueError("alpha_source_mode state_anchor requires production_profile")
        _validate_profile_genes(production_profile, master_genes, profile_gene_policy)
        resolved_graph.validate_states(production_profile.states)
        production_profile.validate_states(list(states))
        if state_master_programs is not None:
            raise ValueError("state_master_programs are only supported when alpha_source_mode continuous_program is active")
    elif production_profile is not None:
        raise ValueError("production_profile requires alpha_source_mode state_anchor")

    grn, grn_calibration = _maybe_calibrate_grn(
        grn,
        master_genes=master_genes,
        production_profile=production_profile,
        auto_calibrate_half_response=auto_calibrate_half_response,
        grn_calibration=grn_calibration,
        target_leak_alpha=target_leak_alpha,
    )

    if resolved_alpha_source_mode == "state_anchor":
        steady_states = _steady_state_table_from_source_rates(
            grn,
            production_profile.rates.reindex(index=list(states), columns=list(master_genes), fill_value=0.0),
            master_genes=master_genes,
            beta=beta_series,
            gamma=gamma_series,
            target_leak_alpha=target_leak_alpha,
        )
    else:
        source_rates = _source_rates_from_program_starts(
            states=states,
            state_offsets=state_offsets,
            total_program_time=total_program_time,
            master_genes=master_genes,
            base_programs=programs,
            state_master_programs=state_master_programs,
        )
        steady_states = _steady_state_table_from_source_rates(
            grn,
            source_rates,
            master_genes=master_genes,
            beta=beta_series,
            gamma=gamma_series,
            target_leak_alpha=target_leak_alpha,
        )

    state_segments: dict[str, dict[str, np.ndarray]] = {}
    state_initialization: dict[str, dict[str, object]] = {}
    for state in states:
        parent_state = resolved_graph.parent_of(state)
        is_root = parent_state is None
        segment_time = float(root_time) if is_root else float(state_durations[state])
        merged_programs = _merge_programs(programs, None if state_master_programs is None else state_master_programs.get(state))

        if resolved_alpha_source_mode == "state_anchor":
            if is_root:
                source_alpha = _source_alpha_from_profile(
                    production_profile,
                    state,
                    grn.genes,
                    master_genes,
                    default_master_alpha,
                    profile_gene_policy,
                )
                source_alpha_fn = None
            else:
                source_alpha = None
                source_alpha_fn = _state_transition_source_alpha_fn(
                    production_profile,
                    parent_state,
                    state,
                    state_offsets[state],
                    segment_time,
                    grn.genes,
                    master_genes,
                    default_master_alpha,
                    profile_gene_policy,
                    transition_schedule,
                    transition_midpoint,
                    transition_steepness,
                )
        else:
            source_alpha = None
            source_alpha_fn = None

        if is_root:
            if u0 is not None or s0 is not None:
                init_u = u0_arr.copy()
                init_s = s0_arr.copy()
                init_source = "explicit_initial_state"
            else:
                init_u = steady_states[state]["u"].reindex(grn.genes).to_numpy(dtype=float)
                init_s = steady_states[state]["s"].reindex(grn.genes).to_numpy(dtype=float)
                init_source = "state_steady_state"
        elif initialization_policy == "parent_steady_state":
            init_u = steady_states[parent_state]["u"].reindex(grn.genes).to_numpy(dtype=float)
            init_s = steady_states[parent_state]["s"].reindex(grn.genes).to_numpy(dtype=float)
            init_source = "parent_state_steady_state"
        else:
            init_u = state_segments[parent_state]["u"][-1].copy()
            init_s = state_segments[parent_state]["s"][-1].copy()
            init_source = "parent_terminal_state"

        state_initialization[state] = {
            "parent_state": parent_state,
            "source": init_source,
            "u0": init_u.copy(),
            "s0": init_s.copy(),
        }
        state_segments[state] = _simulate_segment(
            grn=grn,
            master_genes=master_genes,
            beta=beta_arr,
            gamma=gamma_arr,
            u0=init_u,
            s0=init_s,
            time_end=segment_time,
            dt=dt,
            master_programs=merged_programs,
            default_master_alpha=default_master_alpha,
            target_leak_alpha=target_leak_alpha,
            alpha_max=alpha_max,
            time_offset=state_offsets[state],
            program_time_end=total_program_time,
            source_alpha=source_alpha,
            source_alpha_fn=source_alpha_fn,
            regulator_activity=regulator_activity,
            return_edge_contributions=return_edge_contributions,
        )

    rng = np.random.default_rng(seed)
    sampled_segments: list[tuple[str, dict[str, np.ndarray], np.ndarray]] = []
    sampling_summary: dict[str, dict[str, int | bool]] = {}
    for state in states:
        indices = np.arange(state_segments[state]["u"].shape[0], dtype=int)
        sampled_idx, summary = _sample_snapshots(
            rng,
            int(state_cell_counts[state]),
            indices,
            allow_snapshot_replacement=allow_snapshot_replacement,
        )
        sampled_segments.append((state, state_segments[state], sampled_idx))
        sampling_summary[state] = summary

    true_u = np.concatenate([segment["u"][idx] for _, segment, idx in sampled_segments], axis=0)
    true_s = np.concatenate([segment["s"][idx] for _, segment, idx in sampled_segments], axis=0)
    true_v = np.concatenate([segment["velocity"][idx] for _, segment, idx in sampled_segments], axis=0)
    true_velocity_u = np.concatenate([segment["true_velocity_u"][idx] for _, segment, idx in sampled_segments], axis=0)
    true_alpha = np.concatenate([segment["alpha"][idx] for _, segment, idx in sampled_segments], axis=0)
    resolved_capture_model = _resolve_capture_model_name(capture_model)
    observed = _observed_from_true(
        true_u,
        true_s,
        seed=seed,
        noise_seed=noise_seed,
        capture_rate=capture_rate,
        poisson_observed=poisson_observed,
        dropout_rate=dropout_rate,
        capture_model=resolved_capture_model,
    )

    global_index_map: dict[str, np.ndarray] = {}
    cursor = 0
    for state in states:
        length = state_segments[state]["u"].shape[0]
        global_index_map[state] = np.arange(cursor, cursor + length, dtype=int)
        cursor += length

    obs_frames = []
    offset = 0
    for state, segment, idx in sampled_segments:
        n = len(idx)
        parent_state = resolved_graph.parent_of(state)
        global_idx = global_index_map[state][idx]
        obs_frames.append(
            pd.DataFrame(
                {
                    "pseudotime": segment["pseudotime"][idx],
                    "local_time": segment["local_time"][idx],
                    "branch": state,
                    "segment": state,
                    "state": state,
                    "parent_state": parent_state,
                    "state_depth": state_depths[state],
                    "edge_id": f"{parent_state}->{state}" if parent_state is not None else f"root->{state}",
                    "segment_time_index": idx,
                    "global_time_index": global_idx,
                    "time_index": idx,
                },
                index=[f"cell_{i}" for i in range(offset, offset + n)],
            )
        )
        offset += n
    obs = pd.concat(obs_frames, axis=0)

    time_grid = pd.concat(
        [
            pd.DataFrame(
                {
                    "pseudotime": state_segments[state]["pseudotime"],
                    "local_time": state_segments[state]["local_time"],
                    "branch": state,
                    "state": state,
                    "parent_state": resolved_graph.parent_of(state),
                    "state_depth": state_depths[state],
                    "global_time_index": global_index_map[state],
                }
            )
            for state in states
        ],
        ignore_index=True,
    )
    total_cells = int(obs.shape[0])
    config = {
        "model": "graph_ode_mvp",
        "simulation_mode": simulation_mode,
        "simulator": "graph",
        "root_time": root_time,
        "state_time": state_durations,
        "dt": dt,
        "n_cells": total_cells,
        "n_cells_per_state": state_cell_counts,
        "state_order": list(states),
        "root_states": list(root_states),
        "graph_edges": resolved_graph.edges.to_dict(orient="records"),
        "seed": seed,
        "integrator": "rk4",
        "alpha_max": alpha_max,
        "default_master_alpha": default_master_alpha,
        "explicit_master_regulators": list(master_genes),
        "resolved_master_regulator_source": master_metadata["resolved_master_regulator_source"],
        "incoming_edges_to_masters_count": master_metadata["incoming_edges_to_masters_count"],
        "incoming_edges_to_masters": master_metadata["incoming_edges_to_masters"],
        "allow_profile_targets_as_masters": allow_profile_targets_as_masters,
        "alpha_source_mode": resolved_alpha_source_mode,
        "master_programs": _serialize_master_programs(programs),
        "state_master_programs_enabled": state_master_programs is not None,
        "production_profile": production_profile is not None,
        "production_profile_states": list(production_profile.states) if production_profile is not None else None,
        "production_profile_master_genes": list(production_profile.genes) if production_profile is not None else None,
        "profile_gene_policy": profile_gene_policy,
        "initialization_policy": initialization_policy,
        "sampling_policy": sampling_policy,
        "transition_schedule": transition_schedule if resolved_alpha_source_mode == "state_anchor" else None,
        "transition_midpoint": transition_midpoint if resolved_alpha_source_mode == "state_anchor" else None,
        "transition_steepness": transition_steepness if resolved_alpha_source_mode == "state_anchor" else None,
        "auto_calibrate_half_response": auto_calibrate_half_response,
        "target_leak_alpha": "vector" if not np.isscalar(target_leak_alpha) else float(target_leak_alpha),
        "capture_rate": capture_rate,
        "capture_model": resolved_capture_model,
        "regulator_activity": regulator_activity,
        "poisson_observed": poisson_observed,
        "dropout_rate": dropout_rate,
        "return_edge_contributions": return_edge_contributions,
        "sampling_replace": any(item["sampling_replace"] for item in sampling_summary.values()),
        "n_unique_timepoints_sampled": int(sum(item["n_unique_timepoints_sampled"] for item in sampling_summary.values())),
        "n_duplicate_snapshot_cells": int(sum(item["n_duplicate_snapshot_cells"] for item in sampling_summary.values())),
        "sampling_summary": sampling_summary,
        "time_index_scope": "segment_local",
    }
    noise_config = {
        "capture_model": config["capture_model"],
        "capture_rate": capture_rate,
        "poisson_observed": poisson_observed,
        "dropout_rate": dropout_rate,
    }

    sampled_edge_contributions = None
    if return_edge_contributions:
        sampled_edge_contributions = np.concatenate(
            [segment["edge_contributions"][idx] for _, segment, idx in sampled_segments],
            axis=0,
        )

    result = make_result_dict(
        true_unspliced=true_u,
        true_spliced=true_s,
        true_velocity=true_v,
        velocity_u=true_velocity_u,
        true_alpha=true_alpha,
        observed_unspliced=observed["unspliced"],
        observed_spliced=observed["spliced"],
        obs=obs,
        var=_gene_metadata(grn, master_genes, beta=beta_series, gamma=gamma_series),
        grn=grn,
        beta=beta_series,
        gamma=gamma_series,
        simulation_config=config,
        grn_calibration=_grn_calibration_summary(grn, master_genes, grn_calibration=grn_calibration),
        noise_config=noise_config,
        time_grid=time_grid,
        edge_contributions=sampled_edge_contributions,
        edge_metadata=grn.to_dataframe() if sampled_edge_contributions is not None else None,
    )
    result["uns"]["segment_time_courses"] = state_segments
    result["uns"]["graph"] = {
        "states": list(states),
        "root_states": list(root_states),
        "edges": resolved_graph.edges.to_dict(orient="records"),
        "state_depths": state_depths,
    }
    result["uns"]["state_steady_states"] = {
        state: {
            "alpha": steady_states[state]["alpha"].to_numpy(dtype=float),
            "u": steady_states[state]["u"].to_numpy(dtype=float),
            "s": steady_states[state]["s"].to_numpy(dtype=float),
        }
        for state in states
    }
    result["uns"]["state_initialization"] = state_initialization
    return result


def simulate(
    grn: GRN,
    *,
    simulator: str | None = None,
    simulation_mode: str | None = None,
    graph: StateGraph | pd.DataFrame | dict[str, object] | None = None,
    initialization_policy: str | None = None,
    sampling_policy: str | None = None,
    **kwargs: object,
) -> dict:
    resolved_mode = _resolve_mode_defaults(
        simulator=simulator,
        simulation_mode=simulation_mode,
        graph=graph,
        production_profile=kwargs.get("production_profile"),
        alpha_source_mode=kwargs.get("alpha_source_mode"),
        initialization_policy=initialization_policy,
        sampling_policy=sampling_policy,
        transition_schedule=kwargs.get("transition_schedule"),
        regulator_activity=kwargs.get("regulator_activity"),
        auto_calibrate_half_response=kwargs.get("auto_calibrate_half_response"),
    )
    patched_kwargs = dict(kwargs)
    patched_kwargs["alpha_source_mode"] = resolved_mode["alpha_source_mode"]
    patched_kwargs["regulator_activity"] = resolved_mode["regulator_activity"]
    patched_kwargs["auto_calibrate_half_response"] = resolved_mode["auto_calibrate_half_response"]
    patched_kwargs["transition_schedule"] = resolved_mode["transition_schedule"]
    return _simulate_graph_impl(
        grn,
        graph=resolved_mode["graph"],
        initialization_policy=str(resolved_mode["initialization_policy"]),
        sampling_policy=str(resolved_mode["sampling_policy"]),
        simulation_mode=resolved_mode["simulation_mode"],
        **patched_kwargs,
    )
