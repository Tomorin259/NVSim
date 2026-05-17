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

These are the supported public example workflows.

### Core entrypoints

- `tutorial.py`: the canonical end-to-end tutorial for the current public API.
- `run_sergio_ds6_dynamic_graph_stepfix.py`: canonical DS6 benchmark rerun with the corrected `step` semantics, official-style scanpy/scVelo outputs, and regulation diagnostics.
- `run_ds6_stepfix_observation_compare.py`: clean vs tuned-noisy DS6 observation comparison with scVelo moments phase portraits and total-expression UMAPs.
- `scvelo/run_ds6_stepfix_noisy_dynamic.py`: downstream scVelo dynamical workflow starting from the canonical DS6 noisy raw counts.

## Recommended Check Commands

```bash
python examples/tutorial.py
python examples/run_sergio_ds6_dynamic_graph_stepfix.py
python examples/run_ds6_stepfix_observation_compare.py
python examples/scvelo/run_ds6_stepfix_noisy_dynamic.py
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

- `examples/outputs/tutorial/`: tutorial outputs.
- `examples/outputs/ds6_pt_s3_c300_stepfix/`: canonical DS6 benchmark rerun with corrected `step` transitions and full diagnostics.
- `examples/outputs/ds6_pt_s3_c300_stepfix/obs_compare_tuned/`: canonical clean vs tuned-noisy observation comparison for DS6 stepfix.

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
