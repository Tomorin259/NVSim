# Alpha Source Modes

NVSim supports two ways to provide alpha for master regulators. Both modes keep
the same downstream deterministic RNA velocity ODE:

```text
du_i/dt = alpha_i(t) - beta_i * u_i(t)
ds_i/dt = beta_i * u_i(t) - gamma_i * s_i(t)
true_velocity_i(t) = ds_i/dt
```

Only the source/master-regulator alpha layer changes. Non-master genes still use
the SERGIO-style additive Hill regulation model:

```text
alpha_i(t) = target_leak_alpha_i + sum_j K_ji * H_ji(activity_j(t))
```

## continuous_program

`continuous_program` is the original NVSim mode. Each master regulator has a
continuous time program:

```text
alpha_m(t) = f_m(t)
```

Supported program types are `constant`, `linear_increase`, `linear_decrease`,
`sigmoid_increase`, and `sigmoid_decrease`.

This mode is useful for clean, controlled pseudotime RNA velocity benchmarks
where the external master-regulator forcing should be explicit and smooth.

Example:

```python
simulate(
    grn,
    graph=path_graph(["early", "late"]),
    alpha_source_mode="continuous_program",
    master_programs={"g0": linear_increase(0.2, 1.2), "g1": 0.8},
    n_cells_per_state=100,
    root_time=3.0,
    state_time=3.0,
)
```

## state_anchor

`state_anchor` is the SERGIO-inspired mode. A `StateProductionProfile` stores a
states x master-regulators production table. Each row is a regulatory anchor,
such as a bin, cell type, progenitor state, or lineage state:

```text
             MR1   MR2   MR3
progenitor   1.0   5.0   0.5
lineage_A    5.0   1.0   0.5
lineage_B    1.0   0.8   5.0
```

A static state uses:

```text
alpha_m = B[state, m]
```

A transition from parent to child state uses:

```text
alpha_m(t) = (1 - w(t)) * B[parent_state, m] + w(t) * B[child_state, m]
```

Supported transition schedules are:

- `step`: hard switch at `transition_midpoint`;
- `linear`: linear interpolation over the segment;
- `sigmoid`: smooth S-shaped interpolation.

`sigmoid` is recommended for differentiation-like trajectories because hard
switching can create expression or embedding discontinuities. `linear` is useful
for simpler sensitivity checks.

`StateProductionProfile` is a simulation design input. Its values can be
manually specified, sampled from low/high ranges, copied from SERGIO-style
cell-type/bin production-rate tables, or estimated from external cluster-level
TF activity. NVSim does not infer these production values from scRNA-seq by
default.

Path example:

```python
simulate(
    grn,
    graph=path_graph(["progenitor", "lineage_A"]),
    alpha_source_mode="state_anchor",
    production_profile=profile,
    transition_schedule="sigmoid",
    n_cells_per_state=100,
    root_time=3.0,
    state_time=3.0,
)
```

Branching example:

```python
simulate(
    grn,
    graph=branching_graph("progenitor", ["lineage_A", "lineage_B"]),
    alpha_source_mode="state_anchor",
    production_profile=profile,
    transition_schedule="sigmoid",
    n_cells_per_state={"progenitor": 120, "lineage_A": 100, "lineage_B": 100},
    root_time=3.0,
    state_time={"progenitor": 3.0, "lineage_A": 4.0, "lineage_B": 4.0},
)
```

For graph simulation, `production_profile.index` must cover graph states, and each edge uses `transition_schedule` to define the parent-to-child regulatory-anchor transition. `path_graph(...)` and `branching_graph(...)` are convenience helpers; arbitrary rooted DAGs can be passed through `StateGraph`.

By default, production-profile columns must exactly match the resolved master
regulators (`profile_gene_policy="exact"`). For larger real networks, use
`profile_gene_policy="subset_fill"` to provide anchors for only part of the
master regulators; missing master regulators receive `default_master_alpha`.
Extra non-master columns remain an error.

## Relation To SERGIO

This mode borrows SERGIO's state/bin-specific master-regulator production idea:
different states can have different source-gene production rates while the GRN
topology and edge parameters remain fixed. This adds structured biological
heterogeneity through the source layer.

NVSim does not implement SERGIO's CLE/SDE stochastic simulator, promoter
switching, or SERGIO technical noise. The backend remains deterministic
RNA-velocity ODE integration.
