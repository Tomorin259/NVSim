# NVSim Current Status: v0.1 MVP Checkpoint

## Project Goal

NVSim is a lightweight GRN-aware RNA velocity simulator for controlled method-development and benchmarking experiments. The v0.1 MVP focuses on a transparent chain:

```text
GRN -> alpha(t) -> unspliced/spliced ODE -> true velocity -> snapshot cells -> observed layers -> quick-look plots
```

The implementation is intentionally smaller than SERGIO, VeloSim, dyngen, or scVelo. It provides ground-truth alpha, unspliced/spliced states, velocity, GRN, kinetic parameters, and trajectory metadata.

## Implemented Modules

- `nvsim/grn.py`: validated GRN edge schema, sign normalization, non-negative edge weights, default Hill parameters.
- `nvsim/regulation.py`: Hill activation/repression functions and GRN-controlled alpha computation.
- `nvsim/programs.py`: master regulator alpha programs: constant, linear increase/decrease, sigmoid increase/decrease.
- `nvsim/kinetics.py`: beta/gamma vector creation or validation and non-negative initial `u0/s0` setup.
- `nvsim/trajectory.py`: simple metadata builders for linear and bifurcation trajectories.
- `nvsim/simulate.py`: RK4 ODE integration, linear simulation, bifurcation simulation, snapshot sampling, true and observed layer assembly.
- `nvsim/noise.py`: observed layer generation with capture scaling, optional Poisson sampling, and optional dropout.
- `nvsim/output.py`: plain dictionary output and optional AnnData export.
- `nvsim/plotting.py`: matplotlib-only PCA/UMAP embeddings, velocity quick-look arrows, phase portraits, gene dynamics, and bifurcation representative-gene selection by alpha divergence.
- `nvsim/config.py`: lightweight dataclass configuration defaults.

## Current Modeling Chain

For gene `i`:

```text
du_i/dt = alpha_i(t) - beta_i * u_i(t)
ds_i/dt = beta_i * u_i(t) - gamma_i * s_i(t)
v_i(t) = beta_i * u_i(t) - gamma_i * s_i(t)
```

- `beta_i` and `gamma_i` are gene-specific vectors.
- Master regulator alpha programs are applied only to genes without incoming GRN edges.
- Target-gene `alpha_i(t)` is recomputed from current regulator spliced expression `s_j(t)`, edge weight, edge sign, Hill coefficient, and threshold.
- Activation contribution is `weight * H_act(s_j)`.
- Repression contribution is `weight * H_rep(s_j)`, with no negative sign.
- `u` and `s` are kept non-negative after integration steps.

## Supported Simulations

### Linear

`simulate_linear` integrates one trajectory segment on a fixed time grid with RK4. Snapshot cells are sampled from the simulated grid and returned in cells x genes layers.

### Bifurcation

`simulate_bifurcation` implements a trunk-to-two-branch MVP:

1. Simulate trunk first.
2. Copy terminal trunk `u` and `s` states.
3. Simulate `branch_0` and `branch_1` independently from copied states.
4. Sample and concatenate cells in this order: trunk, `branch_0`, `branch_1`.
5. Keep `pseudotime`, `local_time`, `branch`, `alpha`, `u`, `s`, and velocity aligned.

Branch-specific master regulator programs can be supplied through `branch_master_programs`. If omitted, both branches share the same programs and may remain identical after inheriting the same state.

## Supported Outputs

The default output is a plain Python dictionary with:

- `layers["unspliced"]`
- `layers["spliced"]`
- `layers["true_unspliced"]`
- `layers["true_spliced"]`
- `layers["true_velocity"]`
- `layers["true_alpha"]`
- `obs["pseudotime"]`
- `obs["local_time"]`
- `obs["branch"]`
- `var["gene_class"]`
- `uns["true_grn"]`
- `uns["kinetic_params"]`
- `uns["simulation_config"]`
- `time_grid`

Bifurcation dictionaries additionally include:

- `uns["segment_time_courses"]`
- `uns["branch_inheritance"]`

Layer matrices are cells x genes. Internal segment time courses are timepoints x genes.

Optional AnnData export includes the required layers, obs columns, var metadata, and uns fields when `anndata` is installed.

## Supported Plots

`nvsim/plotting.py` supports:

- PCA embedding from true or observed spliced layers.
- UMAP from PCA coordinates when `umap-learn` is installed; otherwise PCA fallback.
- Embedding colored by pseudotime.
- Embedding colored by branch.
- PCA velocity quick-look arrows using projected true velocity.
- Phase portraits for true or observed layers.
- Gene dynamics over pseudotime for `true_alpha`, `true_unspliced`, `true_spliced`, and `true_velocity`.
- Representative bifurcation gene selection by post-branch `true_alpha` divergence.

True-layer plots are the primary scientific validation views. Observed plots are technical-noise diagnostics. `observed_lownoise` plots use `poisson_observed=False` for visualization/debugging.

## Current Examples

- `examples/run_mvp_linear.py`: generate a 20-gene linear dataset and save `mvp_linear.h5ad` if AnnData is available.
- `examples/plot_linear.py`: generate organized quick-look plots under `examples/plots_linear/`.
- `examples/run_mvp_bifurcation.py`: generate a trunk/two-branch dataset and save `mvp_bifurcation.h5ad` if AnnData is available.
- `examples/plot_bifurcation.py`: generate organized quick-look plots under `examples/plots_bifurcation/` and save selected-gene diagnostics.

## Current Tests

Current tests cover:

- GRN schema validation, sign normalization, default Hill parameters, non-negative weights, unknown-gene rejection.
- Hill activation/repression behavior and repression contribution without negative sign.
- Kinetic and simulation output dimensions.
- Alpha, unspliced, and spliced non-negativity.
- True velocity formula.
- Linear seed reproducibility.
- True/observed layer separation.
- Optional AnnData export fields.
- Bifurcation trunk-to-branch inherited state.
- Independent branch evolution under branch-specific programs.
- Bifurcation metadata alignment.
- `poisson=False` continuous observed layers and `poisson=True` count-like observed layers.
- Plotting smoke tests and dynamic representative-gene selection.
- Constant-alpha ODE steady-state sanity check.

## Known Limitations

- Branch-specific behavior is currently driven by master regulator programs, not formal gene classes.
- No formal normal/MURK/branching gene class system yet.
- No MURK switch-like program model yet.
- No promoter on/off switching.
- No molecule-level SSA.
- No protein or translation layer.
- No SERGIO CLE implementation.
- No VeloSim EVF-to-kinetics mapping.
- Noise is a minimal capture/Poisson/dropout model, not calibrated full UMI realism.
- UMAP is qualitative and can fragment sparse/noisy toy data.
- Velocity arrows are PCA quick-look diagnostics, not a full scVelo-style velocity embedding.
- Large-scale benchmark dataset generation is not implemented.

## Explicit Non-Goals for v0.1

- normal/MURK/branching gene classes.
- promoter switching.
- SERGIO CLE.
- VeloSim EVF-to-kinetics.
- full scVelo-style velocity embedding.
- realistic full UMI noise calibration.
- large-scale benchmark dataset generation.
