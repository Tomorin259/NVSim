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

- `run_linear_continuous_program.py`: canonical linear example for
  `alpha_source_mode="continuous_program"`.
- `run_bifurcation_state_anchor.py`: canonical bifurcation example for
  `alpha_source_mode="state_anchor"`.
- `run_mvp_linear.py`: compact 20-gene linear smoke example.
- `run_mvp_bifurcation.py`: compact trunk-to-two-branch smoke example with
  continuous master-regulator programs.
- `run_trrust_mouse_small.py`: external TRRUST signed-GRN benchmark example.

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
python examples/run_bifurcation_state_anchor.py
python examples/plot_velocity_showcase.py
python examples/run_trrust_mouse_small.py
```

## Output Naming

Current generated output groups are:

- `examples/outputs/tutorial/`: tutorial linear and bifurcation outputs.
- `examples/outputs/linear_continuous_program/`: canonical linear continuous
  master-program example.
- `examples/outputs/bifurcation_state_anchor/`: canonical bifurcation
  state-anchor example.
- `examples/outputs/linear_20gene/`: small linear MVP sanity dataset.
- `examples/outputs/bifurcation_20gene_3master/`: small hand-built GRN
  bifurcation dataset.
- `examples/outputs/velocity_showcase/`: scanpy/scVelo velocity-style
  showcase plots.
- `examples/outputs/trrust_mouse_small/`: external TRRUST benchmark outputs.
