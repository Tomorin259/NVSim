"""NVSim MVP 的核心 ODE 模拟器。

本模块把完整建模链串起来：GRN -> alpha(t) -> unspliced u(t) ->
spliced s(t) -> true velocity。核心方程是：

    du/dt = alpha(t) - beta * u
    ds/dt = beta * u - gamma * s

其中 true_velocity 保存 ds/dt，true_velocity_u 保存 du/dt。
线性轨迹和 bifurcation 都复用同一个 segment 积分器。
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .grn import GRN
from .kinetics import create_kinetic_vectors, initialize_state
from .noise import generate_observed_counts
from .output import make_result_dict
from .programs import AlphaProgram, coerce_programs, constant
from .regulation import compute_alpha


def _master_genes(grn: GRN) -> tuple[str, ...]:
    """返回没有 incoming edges 的基因，即需要外部 alpha program 的 master。"""

    incoming = set(grn.edges["target"])
    return tuple(gene for gene in grn.genes if gene not in incoming)


def _alpha_from_state(
    s: np.ndarray,
    t: float,
    time_end: float,
    grn: GRN,
    master_programs: Mapping[str, AlphaProgram],
    default_master_alpha: float,
    alpha_max: float | None,
) -> np.ndarray:
    # 根据当前 spliced 状态 s(t) 重新计算 alpha(t)。
    # master 的 alpha 来自时间程序；target/intermediate 的 alpha 来自 GRN。
    genes = grn.genes
    normalized_t = 0.0 if time_end == 0 else float(np.clip(t / time_end, 0.0, 1.0))
    # basal 先全设为 0；只有 master gene 会写入外部 alpha program。
    # target gene 的 alpha 不直接指定，而由 compute_alpha 累加 GRN 贡献。
    basal = pd.Series(0.0, index=pd.Index(genes, name="gene"), dtype=float)
    for gene in _master_genes(grn):
        program = master_programs.get(gene, constant(default_master_alpha))
        basal.loc[gene] = program.value(normalized_t)
    # MVP 假设 regulator activity = 当前 spliced RNA s_j(t)。
    regulator_values = pd.Series(np.maximum(s, 0.0), index=pd.Index(genes, name="gene"), dtype=float)
    alpha = compute_alpha(regulator_values, grn, basal_alpha=basal, alpha_min=0.0, alpha_max=alpha_max)
    return alpha.reindex(genes).to_numpy(dtype=float)


def _derivative(
    y: np.ndarray,
    t: float,
    time_end: float,
    grn: GRN,
    beta: np.ndarray,
    gamma: np.ndarray,
    master_programs: Mapping[str, AlphaProgram],
    default_master_alpha: float,
    alpha_max: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    # y 是拼接状态向量：[u_1...u_G, s_1...s_G]。
    n_genes = len(beta)
    u = np.maximum(y[:n_genes], 0.0)
    s = np.maximum(y[n_genes:], 0.0)
    alpha = _alpha_from_state(s, t, time_end, grn, master_programs, default_master_alpha, alpha_max)
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
    beta: np.ndarray,
    gamma: np.ndarray,
    master_programs: Mapping[str, AlphaProgram],
    default_master_alpha: float,
    alpha_max: float | None,
) -> np.ndarray:
    """执行一步 RK4；每个中间状态都会重新通过 GRN 计算 alpha。"""

    k1, _ = _derivative(y, t, time_end, grn, beta, gamma, master_programs, default_master_alpha, alpha_max)
    k2, _ = _derivative(
        y + 0.5 * dt * k1,
        t + 0.5 * dt,
        time_end,
        grn,
        beta,
        gamma,
        master_programs,
        default_master_alpha,
        alpha_max,
    )
    k3, _ = _derivative(
        y + 0.5 * dt * k2,
        t + 0.5 * dt,
        time_end,
        grn,
        beta,
        gamma,
        master_programs,
        default_master_alpha,
        alpha_max,
    )
    k4, _ = _derivative(y + dt * k3, t + dt, time_end, grn, beta, gamma, master_programs, default_master_alpha, alpha_max)
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
    beta: np.ndarray,
    gamma: np.ndarray,
    u0: np.ndarray,
    s0: np.ndarray,
    time_end: float,
    dt: float,
    master_programs: Mapping[str, AlphaProgram],
    default_master_alpha: float,
    alpha_max: float | None,
    time_offset: float = 0.0,
    program_time_end: float | None = None,
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

    u[0] = np.maximum(u0, 0.0)
    s[0] = np.maximum(s0, 0.0)
    alpha[0] = _alpha_from_state(s[0], global_time[0], alpha_time_end, grn, master_programs, default_master_alpha, alpha_max)
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
            beta,
            gamma,
            master_programs,
            default_master_alpha,
            alpha_max,
        )
        u[step] = y[:n_genes]
        s[step] = y[n_genes:]
        alpha[step] = _alpha_from_state(s[step], global_time[step], alpha_time_end, grn, master_programs, default_master_alpha, alpha_max)
        velocity[step] = beta * u[step] - gamma * s[step]
        true_velocity_u[step] = alpha[step] - beta * u[step]

    return {
        "local_time": local_time,
        "pseudotime": global_time,
        "actual_dt": np.asarray(actual_dt),
        "u": u,
        "s": s,
        "alpha": alpha,
        "velocity": velocity,
        "true_velocity_u": true_velocity_u,
    }


def _sample_snapshots(
    rng: np.random.Generator,
    n_cells: int,
    n_timepoints: int,
) -> np.ndarray:
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")
    replace = n_cells > n_timepoints
    sampled = rng.choice(np.arange(n_timepoints), size=n_cells, replace=replace)
    return np.sort(sampled)


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
    u0_series, s0_series = initialize_state(genes, u0=u0, s0=s0, seed=seed + 1)
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


def _gene_metadata(grn: GRN) -> pd.DataFrame:
    var = pd.DataFrame(index=pd.Index(grn.genes, name="gene"))
    master_set = set(_master_genes(grn))
    var["gene_class"] = ["master_regulator" if gene in master_set else "target" for gene in grn.genes]
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
) -> dict[str, np.ndarray]:
    return generate_observed_counts(
        true_u,
        true_s,
        seed=seed + 2 if noise_seed is None else noise_seed,
        capture_rate=capture_rate,
        poisson=poisson_observed,
        dropout_rate=dropout_rate,
    )


def _merge_programs(
    base: Mapping[str, AlphaProgram],
    overrides: Mapping[str, AlphaProgram | float] | None,
) -> dict[str, AlphaProgram]:
    merged = dict(base)
    if overrides:
        merged.update(coerce_programs(overrides))
    return merged


def simulate_linear(
    grn: GRN,
    n_cells: int = 100,
    time_end: float = 1.0,
    dt: float = 0.01,
    beta: object | None = None,
    gamma: object | None = None,
    u0: object | None = None,
    s0: object | None = None,
    master_programs: Mapping[str, AlphaProgram | float] | None = None,
    default_master_alpha: float = 0.5,
    alpha_max: float | None = None,
    seed: int = 0,
    noise_seed: int | None = None,
    capture_rate: float | None = None,
    poisson_observed: bool = True,
    dropout_rate: float = 0.0,
) -> dict:
    """Simulate a linear GRN-aware RNA velocity trajectory.

    The ODE is integrated on a dense fixed time grid with RK4. Snapshot cells are
    sampled from that grid, and observed layers are generated separately from the
    true layers. Returned layer matrices are cells x genes; internal time-course
    arrays are timepoints x genes.
    """

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

    segment = _simulate_segment(
        grn=grn,
        beta=beta_arr,
        gamma=gamma_arr,
        u0=u0_arr,
        s0=s0_arr,
        time_end=time_end,
        dt=dt,
        master_programs=programs,
        default_master_alpha=default_master_alpha,
        alpha_max=alpha_max,
        time_offset=0.0,
        program_time_end=time_end,
    )

    # snapshot sampling：先模拟连续时间过程，再抽样为离散细胞。
    rng = np.random.default_rng(seed)
    sampled_idx = _sample_snapshots(rng, n_cells, segment["u"].shape[0])
    true_u = segment["u"][sampled_idx]
    true_s = segment["s"][sampled_idx]
    true_v = segment["velocity"][sampled_idx]
    true_velocity_u = segment["true_velocity_u"][sampled_idx]
    true_alpha = segment["alpha"][sampled_idx]
    observed = _observed_from_true(
        true_u,
        true_s,
        seed=seed,
        noise_seed=noise_seed,
        capture_rate=capture_rate,
        poisson_observed=poisson_observed,
        dropout_rate=dropout_rate,
    )

    obs = pd.DataFrame(
        {
            "pseudotime": segment["pseudotime"][sampled_idx],
            "local_time": segment["local_time"][sampled_idx],
            "branch": "linear",
            "time_index": sampled_idx,
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    time_grid = pd.DataFrame({"time": segment["pseudotime"], "local_time": segment["local_time"], "branch": "linear"})
    config = {
        "model": "linear_ode_mvp",
        "time_end": time_end,
        "dt": dt,
        "actual_dt": float(segment["actual_dt"]),
        "n_timepoints": int(segment["u"].shape[0]),
        "n_cells": n_cells,
        "seed": seed,
        "integrator": "rk4",
        "alpha_max": alpha_max,
        "default_master_alpha": default_master_alpha,
        "capture_rate": capture_rate,
        "poisson_observed": poisson_observed,
        "dropout_rate": dropout_rate,
    }

    return make_result_dict(
        true_unspliced=true_u,
        true_spliced=true_s,
        true_velocity=true_v,
        velocity_u=true_velocity_u,
        true_alpha=true_alpha,
        observed_unspliced=observed["unspliced"],
        observed_spliced=observed["spliced"],
        obs=obs,
        var=_gene_metadata(grn),
        grn=grn,
        beta=beta_series,
        gamma=gamma_series,
        simulation_config=config,
        time_grid=time_grid,
    )


def simulate_bifurcation(
    grn: GRN,
    n_trunk_cells: int = 50,
    n_branch_cells: int | Mapping[str, int] = 60,
    trunk_time: float = 2.0,
    branch_time: float = 2.0,
    dt: float = 0.01,
    beta: object | None = None,
    gamma: object | None = None,
    u0: object | None = None,
    s0: object | None = None,
    master_programs: Mapping[str, AlphaProgram | float] | None = None,
    branch_master_programs: Mapping[str, Mapping[str, AlphaProgram | float]] | None = None,
    default_master_alpha: float = 0.5,
    alpha_max: float | None = None,
    seed: int = 0,
    noise_seed: int | None = None,
    capture_rate: float | None = None,
    poisson_observed: bool = True,
    dropout_rate: float = 0.0,
) -> dict:
    """Simulate a trunk-to-two-branch RNA velocity trajectory.

    The trunk is simulated first. The terminal trunk ``u`` and ``s`` vectors are
    copied into each branch initial state, then each branch is integrated as an
    independent segment. Returned layer matrices are cells x genes; internal
    ``uns["segment_time_courses"]`` arrays are timepoints x genes. If no
    ``branch_master_programs`` are supplied, both branches share the same master
    regulator programs and may follow identical dynamics from the inherited
    state.
    """

    branch_labels = ("branch_0", "branch_1")
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

    total_program_time = trunk_time + branch_time
    trunk = _simulate_segment(
        grn=grn,
        beta=beta_arr,
        gamma=gamma_arr,
        u0=u0_arr,
        s0=s0_arr,
        time_end=trunk_time,
        dt=dt,
        master_programs=programs,
        default_master_alpha=default_master_alpha,
        alpha_max=alpha_max,
        time_offset=0.0,
        program_time_end=total_program_time,
    )
    # bifurcation 的关键：两个 branch 必须继承同一个 trunk terminal state。
    inherited_u = trunk["u"][-1].copy()
    inherited_s = trunk["s"][-1].copy()

    branch_segments: dict[str, dict[str, np.ndarray]] = {}
    for branch in branch_labels:
        branch_segments[branch] = _simulate_segment(
            grn=grn,
            beta=beta_arr,
            gamma=gamma_arr,
            u0=inherited_u.copy(),
            s0=inherited_s.copy(),
            time_end=branch_time,
            dt=dt,
            master_programs=_merge_programs(programs, None if branch_master_programs is None else branch_master_programs.get(branch)),
            default_master_alpha=default_master_alpha,
            alpha_max=alpha_max,
            time_offset=trunk_time,
            program_time_end=total_program_time,
        )

    rng = np.random.default_rng(seed)
    sampled_segments: list[tuple[str, dict[str, np.ndarray], np.ndarray]] = []
    trunk_idx = _sample_snapshots(rng, n_trunk_cells, trunk["u"].shape[0])
    sampled_segments.append(("trunk", trunk, trunk_idx))
    for branch in branch_labels:
        count = n_branch_cells[branch] if isinstance(n_branch_cells, Mapping) else n_branch_cells
        branch_idx = _sample_snapshots(rng, int(count), branch_segments[branch]["u"].shape[0])
        sampled_segments.append((branch, branch_segments[branch], branch_idx))

    # 按 trunk, branch_0, branch_1 拼接，保证 layers 与 obs 行顺序一致。
    true_u = np.concatenate([segment["u"][idx] for _, segment, idx in sampled_segments], axis=0)
    true_s = np.concatenate([segment["s"][idx] for _, segment, idx in sampled_segments], axis=0)
    true_v = np.concatenate([segment["velocity"][idx] for _, segment, idx in sampled_segments], axis=0)
    true_velocity_u = np.concatenate([segment["true_velocity_u"][idx] for _, segment, idx in sampled_segments], axis=0)
    true_alpha = np.concatenate([segment["alpha"][idx] for _, segment, idx in sampled_segments], axis=0)
    observed = _observed_from_true(
        true_u,
        true_s,
        seed=seed,
        noise_seed=noise_seed,
        capture_rate=capture_rate,
        poisson_observed=poisson_observed,
        dropout_rate=dropout_rate,
    )

    obs_frames = []
    offset = 0
    for branch, segment, idx in sampled_segments:
        n = len(idx)
        obs_frames.append(
            pd.DataFrame(
                {
                    "pseudotime": segment["pseudotime"][idx],
                    "local_time": segment["local_time"][idx],
                    "branch": branch,
                    "segment": branch,
                    "time_index": idx,
                },
                index=[f"cell_{i}" for i in range(offset, offset + n)],
            )
        )
        offset += n
    obs = pd.concat(obs_frames, axis=0)

    time_grid = pd.concat(
        [
            pd.DataFrame({"pseudotime": trunk["pseudotime"], "local_time": trunk["local_time"], "branch": "trunk"}),
            *[
                pd.DataFrame(
                    {
                        "pseudotime": branch_segments[branch]["pseudotime"],
                        "local_time": branch_segments[branch]["local_time"],
                        "branch": branch,
                    }
                )
                for branch in branch_labels
            ],
        ],
        ignore_index=True,
    )
    total_cells = int(obs.shape[0])
    config = {
        "model": "bifurcation_ode_mvp",
        "trunk_time": trunk_time,
        "branch_time": branch_time,
        "dt": dt,
        "actual_dt_trunk": float(trunk["actual_dt"]),
        "actual_dt_branch_0": float(branch_segments["branch_0"]["actual_dt"]),
        "actual_dt_branch_1": float(branch_segments["branch_1"]["actual_dt"]),
        "n_trunk_cells": n_trunk_cells,
        "n_branch_cells": dict(n_branch_cells) if isinstance(n_branch_cells, Mapping) else {b: int(n_branch_cells) for b in branch_labels},
        "n_cells": total_cells,
        "branch_labels": list(branch_labels),
        "seed": seed,
        "integrator": "rk4",
        "alpha_max": alpha_max,
        "default_master_alpha": default_master_alpha,
        "branch_master_programs_enabled": branch_master_programs is not None,
        "capture_rate": capture_rate,
        "poisson_observed": poisson_observed,
        "dropout_rate": dropout_rate,
    }

    result = make_result_dict(
        true_unspliced=true_u,
        true_spliced=true_s,
        true_velocity=true_v,
        velocity_u=true_velocity_u,
        true_alpha=true_alpha,
        observed_unspliced=observed["unspliced"],
        observed_spliced=observed["spliced"],
        obs=obs,
        var=_gene_metadata(grn),
        grn=grn,
        beta=beta_series,
        gamma=gamma_series,
        simulation_config=config,
        time_grid=time_grid,
    )
    result["uns"]["segment_time_courses"] = {
        "trunk": trunk,
        "branch_0": branch_segments["branch_0"],
        "branch_1": branch_segments["branch_1"],
    }
    result["uns"]["branch_inheritance"] = {
        "trunk_terminal_u": inherited_u.copy(),
        "trunk_terminal_s": inherited_s.copy(),
        "branch_0_initial_u": branch_segments["branch_0"]["u"][0].copy(),
        "branch_0_initial_s": branch_segments["branch_0"]["s"][0].copy(),
        "branch_1_initial_u": branch_segments["branch_1"]["u"][0].copy(),
        "branch_1_initial_s": branch_segments["branch_1"]["s"][0].copy(),
    }
    return result
