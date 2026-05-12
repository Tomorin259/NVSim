# NVSim v0.1 Validation Report

## Summary

The current NVSim v0.1 MVP satisfies the intended scientific chain:

```text
GRN -> alpha(t) -> unspliced/spliced ODE -> true velocity -> snapshot cells -> observed layers -> quick-look plots
```

The implementation remains intentionally scoped. It supports linear and bifurcation simulations, true and observed layer separation, optional AnnData export, and qualitative plotting diagnostics. It does not yet implement gene classes, promoter switching, SERGIO CLE, VeloSim EVF-to-kinetics, or scVelo-style velocity embedding.

## A. GRN and Regulation

Status: satisfied for v0.1.

- GRN edges use required columns `regulator`, `target`, `weight`, and `sign`.
- Optional `hill_coefficient` and `threshold` are filled from defaults when absent.
- Edge weights are validated as finite and non-negative.
- Edge signs are normalized to `activation` or `repression`.
- Activation uses `weight * H_act(s_j)`.
- Repression uses `weight * H_rep(s_j)` with no negative sign.
- `compute_alpha` accumulates target alpha from all incoming edges.
- During simulation, regulator values are current non-negative spliced states `s_j(t)`.

Relevant tests:

- `tests/test_grn.py`
- `tests/test_regulation.py`
- `tests/test_simulate.py`

## B. Master Regulator Programs

Status: satisfied for v0.1.

- Master regulators are defined as genes without incoming GRN edges.
- `_alpha_from_state` initializes basal alpha only for master regulators.
- Genes with incoming edges receive alpha through GRN regulation and are not overwritten by master programs.
- Target genes may still have basal zero plus incoming regulatory contributions.

Known limitation:

- Branch-specific programs are supported only as simple overrides for master regulator programs. There is no formal gene-class model yet.

## C. ODE and Velocity

Status: satisfied for v0.1.

The implemented ODE is:

```text
du/dt = alpha - beta*u
ds/dt = beta*u - gamma*s
true_velocity = beta*u - gamma*s
```

- RK4 is used for fixed-step integration.
- `_rk4_step` calls `_derivative` at `t`, `t + dt/2`, `t + dt/2`, and `t + dt`.
- `_derivative` recomputes alpha from the intermediate spliced state each time.
- `u` and `s` are clipped to non-negative values after each RK4 step.
- Stored and sampled `true_velocity` is computed from the same `u` and `s` state stored in the time course or sampled layer.

Relevant tests:

- Velocity formula test.
- Non-negativity test.
- Constant-alpha steady-state sanity test.

## D. Linear Simulation

Status: satisfied for v0.1.

- `simulate_linear` runs on a fixed time grid.
- Snapshot cells are sampled reproducibly with a deterministic seed.
- True and observed layers are separated.
- `examples/run_mvp_linear.py` runs and writes `examples/mvp_linear.h5ad` when AnnData is available.
- `examples/plot_velocity_showcase.py` writes scanpy/scVelo velocity showcase plots under `examples/outputs/velocity_showcase/`.
- True-layer UMAP by pseudotime is the primary visual diagnostic for smooth underlying dynamics.
- Phase portraits use gene-level 2D velocity arrows from `true_velocity=ds/dt` and `true_velocity_u=du/dt`.

## E. Bifurcation Simulation

Status: satisfied for v0.1.

- `simulate_bifurcation` simulates the trunk first.
- Terminal trunk `u` and `s` are copied into each branch initial state.
- `branch_0` and `branch_1` are simulated independently after the branch point.
- Sampled cell order is trunk, then `branch_0`, then `branch_1`.
- `pseudotime`, `local_time`, `branch`, `alpha`, `u`, `s`, and velocity are concatenated in the same order.
- Internal segment time courses are exposed under `uns["segment_time_courses"]` for validation.
- Branch inheritance vectors are exposed under `uns["branch_inheritance"]`.

Relevant tests:

- Branch initial states equal trunk terminal state.
- Branches evolve independently when branch-specific programs differ.
- Bifurcation layer shapes and obs metadata alignment.
- Bifurcation reproducibility.

## F. Output

Status: satisfied for v0.1.

- Sampled layer matrices are cells x genes.
- Internal segment arrays are timepoints x genes.
- True layers and observed layers are separate arrays.
- Plain dictionary output includes layers, obs, var, uns, and time_grid.
- AnnData export includes:
  - `adata.layers["unspliced"]`
  - `adata.layers["spliced"]`
  - `adata.layers["true_unspliced"]`
  - `adata.layers["true_spliced"]`
  - `adata.layers["true_velocity"]`
  - `adata.layers["true_alpha"]`
  - `adata.obs["pseudotime"]`
  - `adata.obs["branch"]`
  - `adata.obs["local_time"]` when present in result obs
  - `adata.var["gene_role"]`
  - `adata.var["gene_class"]`
  - `adata.uns["true_grn"]`
  - `adata.uns["kinetic_params"]`
  - `adata.uns["simulation_config"]`

Known limitation:

- Bifurcation internal `segment_time_courses` are kept in the plain dictionary and are not exported into AnnData uns by `to_anndata`.

## G. Noise

Status: satisfied for v0.1.

- Observed layers can use capture scaling.
- Observed layers can use Poisson sampling.
- Observed layers can use dropout.
- `poisson_observed=False` produces continuous captured true values after optional dropout.
- `observed_lownoise` examples use `capture_rate=1.0`, `dropout_rate=0.0`, and `poisson_observed=False`.

Important interpretation:

- Low-noise observed mode is intended for visualization/debugging only.
- It is not realistic UMI modeling.
- The current noise model is intentionally minimal and not calibrated for large benchmark realism.

Relevant tests:

- `tests/test_noise.py`
- True/observed separation tests in `tests/test_simulate.py`

## Metadata Semantics

Status: clarified for v0.1.

- `gene_role` records current GRN role:
  - `master_regulator`
  - `target`
- `gene_class` is intentionally not used for role labels.
- `gene_class` is reserved for future biological class systems and is currently `unassigned` for every gene.

## H. Plotting

Status: satisfied for v0.1 quick-look diagnostics.

- True plots are the primary validation plots.
- Observed plots show technical-noise effects.
- Observed UMAP can fragment sparse toy datasets and should not be overinterpreted.
- PCA velocity arrows are qualitative quick-look diagnostics, not full scVelo-style velocity embedding.
- Phase portraits default to true layers because observed UMI counts are discrete under capture/Poisson noise.
- Representative bifurcation genes are selected by post-branch `true_alpha` divergence when branch labels are present.

Relevant tests:

- Plotting functions save PNGs.
- True and observed PCA layer preferences run.
- Velocity plotting defaults to PCA.
- AnnData plotting works when AnnData is installed.
- Dynamic representative-gene selection returns valid edge types.

## Test Coverage Review

Current tests cover the requested categories:

- GRN validation: yes.
- Hill activation/repression: yes.
- Alpha non-negativity: yes.
- u/s non-negativity: yes.
- Velocity formula: yes.
- Seed reproducibility: yes.
- Output shape cells x genes: yes.
- AnnData export fields: yes.
- Bifurcation branch inheritance: yes.
- `poisson_observed=False` behavior: yes.
- Plotting functions save files: yes.
- Constant-alpha ODE steady-state sanity: yes.

## Remaining Scientific and Engineering Risks

- Branch dynamics are currently simple and controlled by branch-specific master regulator programs, not formal gene classes.
- Strong biological realism should not be inferred from the current toy examples.
- Observed UMAP can be visually unstable under sparse/noisy toy counts.
- Noise is not calibrated to real library-size or gene-specific UMI behavior.
- PCA-projected velocity arrows are useful diagnostics but not a replacement for proper velocity embedding methods.
- The package has no large-scale benchmark-generation API yet.

## Validation Conclusion

NVSim v0.1 is ready to checkpoint as a lightweight MVP. It satisfies the intended core invariants for GRN-controlled alpha, RNA velocity ODE simulation, true/observed layer separation, bifurcation inheritance, reproducibility, optional AnnData export, and quick-look diagnostics. The next scientific feature layer should be formal gene classes, but that is explicitly outside this checkpoint.
