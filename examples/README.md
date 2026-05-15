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

- `run_linear_continuous_program.py`: canonical path-graph example for
  `alpha_source_mode="continuous_program"`.
- `run_branching_state_anchor.py`: canonical branching-graph example for
  `alpha_source_mode="state_anchor"`.
- `run_mvp_linear.py`: compact 20-gene path-graph smoke example.
- `run_mvp_branching.py`: compact branching-graph smoke example with
  continuous master-regulator programs.
- `run_trrust_mouse_small.py`: external TRRUST signed-GRN benchmark example.
- `run_sergio_ds6_dynamic_graph_stepfix.py`: canonical DS6 benchmark rerun with the corrected `step` semantics, official-style scanpy/scVelo outputs, and regulation diagnostics.

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
python examples/run_sergio_ds6_dynamic_graph_stepfix.py
```

## Output Naming

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
- `examples/outputs/ds6_pt_s3_c300_stepfix_kinsergio/`: the same benchmark with fixed SERGIO kinetics for controlled comparison.
