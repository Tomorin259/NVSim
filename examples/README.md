# NVSim Examples

This directory contains the public example workflows that should stay aligned
with the current canonical NVSim API.

Generated files are written under `examples/outputs/` and are intentionally
ignored by `.gitignore` because they can be regenerated.

## Start Here

- `tutorial.py`: the canonical end-to-end tutorial for the current public API.
  It shows the recommended imports, GRN schema, noise settings, and both
  `simulate_linear()` and `simulate_bifurcation()` calls.

If the public simulation interface changes, update `tutorial.py` in the same
change set.

## Core Examples

These are the supported example workflows.

### Data generation

- `run_mvp_linear.py`: builds a 20-gene linear GRN-aware RNA velocity dataset.
- `run_mvp_bifurcation.py`: builds a trunk-to-two-branch dataset with
  continuous master-regulator programs.
- `run_linear_continuous_program.py`: focused example of the
  `continuous_program` master-regulator alpha mode.
- `run_bifurcation_state_anchor.py`: focused example of the `state_anchor`
  interface with `trunk_state`, `branch_child_states`, and
  `transition_schedule`.

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
python examples/run_mvp_linear.py
python examples/run_mvp_bifurcation.py
python examples/run_linear_continuous_program.py
python examples/run_bifurcation_state_anchor.py
python examples/plot_velocity_showcase.py
```

## Output Naming

Current generated output groups are:

- `examples/outputs/tutorial/`: tutorial linear and bifurcation outputs.
- `examples/outputs/linear_20gene/`: small linear MVP sanity dataset.
- `examples/outputs/bifurcation_20gene_3master/`: small hand-built GRN
  bifurcation dataset.
- `examples/outputs/linear_continuous_program/`: continuous master alpha
  program example.
- `examples/outputs/bifurcation_state_anchor/`: state-anchor bifurcation
  example.
- `examples/outputs/velocity_showcase/`: scanpy/scVelo velocity-style
  showcase plots.
