# NVSim Examples

This directory contains the public example workflows that should stay aligned
with the current canonical NVSim API.

Generated files are written under `examples/outputs/` and are intentionally
ignored by `.gitignore` because they can be regenerated.

## Start Here

- `tutorial.py`: the canonical end-to-end tutorial for the current public API.
  It shows the recommended imports, GRN schema, noise settings, and unified
  graph-based `simulate(..., graph=...)` calls for both path-like and branching
  topologies.

If the public simulation interface changes, update `tutorial.py` in the same
change set.

## Core Examples

These are the supported example workflows.

### Data generation

- `prepare_trrust_mouse_dataset.py`: prepares the external TRRUST mouse signed-GRN benchmark dataset under `data/external/trrust_mouse/`.
- `run_linear_continuous_program.py`: canonical path-graph example for
  `alpha_source_mode="continuous_program"`.
- `run_branching_state_anchor.py`: canonical branching-graph example for
  `alpha_source_mode="state_anchor"`.
- `run_mvp_linear.py`: compact 20-gene path-graph smoke example.
- `run_mvp_branching.py`: compact branching-graph smoke example with
  continuous master-regulator programs.
- `run_trrust_mouse_small.py`: external TRRUST signed-GRN benchmark example.
- `run_sergio_ds6_dynamic_graph_stepfix.py`: canonical DS6 benchmark rerun with the corrected `step` semantics, official-style scanpy/scVelo outputs, and regulation diagnostics.
- `run_sergio_1200g_ds3_strong_branching.py`: SERGIO 1200G DS3 benchmark runner on the unified graph-based simulator.
- `run_ds6_stepfix_observation_compare.py`: clean vs tuned-noisy DS6 observation comparison with scVelo moments phase portraits and total-expression UMAPs.

### Plotting

- `plot_velocity_showcase.py`: uses `nvsim.plotting` to generate scanpy/scVelo
  RNA velocity-style showcase figures under
  `examples/outputs/velocity_showcase/`.

`nvsim.plotting` is the single public plotting module. It delegates
PCA/neighbors/UMAP to scanpy and velocity stream visualization to scVelo, while
keeping NVSim-specific diagnostics such as gene dynamics and two-dimensional
gene phase portraits.

## Recommended Check Commands

```bash
python examples/tutorial.py
python examples/run_linear_continuous_program.py
python examples/run_branching_state_anchor.py
python examples/plot_velocity_showcase.py
python examples/run_trrust_mouse_small.py
python examples/prepare_trrust_mouse_dataset.py
python examples/run_sergio_ds6_dynamic_graph_stepfix.py
python examples/run_sergio_1200g_ds3_strong_branching.py
python examples/run_ds6_stepfix_observation_compare.py
```

## Output Naming

For the canonical DS6 stepfix outputs, file names now encode the processing stage explicitly:

- `ds6_stepfix_clean_simulation.h5ad`: canonical clean simulator output written directly by `run_sergio_ds6_dynamic_graph_stepfix.py`.
- `ds6_stepfix_obs_clean_raw.h5ad`: clean observed `spliced` / `unspliced` counts before scVelo preprocessing.
- `ds6_stepfix_obs_noisy_raw.h5ad`: noisy observed `spliced` / `unspliced` counts before scVelo preprocessing.
- `ds6_stepfix_obs_clean_scvelo_moments.h5ad`: clean observed counts after `filter_and_normalize + moments`.
- `ds6_stepfix_obs_noisy_scvelo_moments.h5ad`: noisy observed counts after `filter_and_normalize + moments`.
- `ds6_stepfix_obs_clean_total_umap.h5ad`: clean total-expression UMAP working object.
- `ds6_stepfix_obs_noisy_total_umap.h5ad`: noisy total-expression UMAP working object.
- `ds6_stepfix_obs_noisy_scvelo_dynamical.h5ad`: scVelo dynamical output starting from `ds6_stepfix_obs_noisy_raw.h5ad`.

Plot files follow the same rule:

- `phase_*_scvelo_moments_*.png`: phase portraits built from `Mu/Ms`.
- `umap_*_total_*.png`: total-expression UMAPs.
- `umap_scvelo_*.png`: scVelo dynamical UMAPs and stream plots.


Current generated output groups are:

- `examples/outputs/tutorial/`: tutorial path-graph and branching-graph outputs.
- `examples/outputs/linear_continuous_program/`: canonical path-graph continuous
  master-program example.
- `examples/outputs/branching_state_anchor/`: canonical branching-graph
  state-anchor example.
- `examples/outputs/linear_20gene/`: small path-graph MVP sanity dataset.
- `examples/outputs/branching_20gene_3master/`: small hand-built GRN
  branching-graph dataset.
- `examples/outputs/velocity_showcase/`: scanpy/scVelo velocity-style
  showcase plots.
- `examples/outputs/trrust_mouse_small/`: external TRRUST benchmark outputs.
- `examples/outputs/ds6_pt_s3_c300_stepfix/`: canonical DS6 benchmark rerun with corrected `step` transitions and full diagnostics.
- `examples/outputs/ds6_pt_s3_c300_stepfix/obs_compare_tuned/`: canonical clean vs tuned-noisy observation comparison for DS6 stepfix.
- `examples/outputs/ds6_pt_s3_c300_stepfix_kinsergio/`: the same benchmark with fixed SERGIO kinetics for controlled comparison.

## DS6 Canonical vs Legacy

Within `examples/outputs/ds6_pt_s3_c300_stepfix/`:

- Canonical reusable artifacts:
  - `ds6_stepfix_clean_simulation.h5ad`
  - `official_style_scanpy_scvelo/`
  - `regulation_diagnostics/`
  - `velocity_showcase/`
  - `obs_compare_tuned/`
- Legacy exploratory directories kept only for historical comparison:
  - `phase_portrait_utils_scvelo_moments_clean_vs_noisy/`
  - `phase_portrait_utils_scvelo_moments_clean_vs_noisy_tuned/`

Prefer the canonical directories above for all new analysis and scripts.
