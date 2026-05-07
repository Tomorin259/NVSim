# NVSim Examples

This directory contains the reproducible example workflows that are expected to work from a fresh clone of the public repository.
Generated files are written under `outputs/` and are intentionally ignored by `.gitignore` because they can be regenerated.

## Core Examples

These are the supported public example workflows.

### Data generation

- `run_mvp_linear.py`: builds a 20-gene linear GRN-aware RNA velocity dataset.
- `run_mvp_bifurcation.py`: builds a trunk-to-two-branch dataset with inherited branch initial states.

### Plotting

- `plot_linear.py`: generates quick-look figures under `outputs/linear_20gene/plots/`.
- `plot_bifurcation.py`: generates quick-look figures under `outputs/bifurcation_20gene_3master/plots/` and writes `diagnostics/selected_genes.txt`.

## Recommended Check Commands

```bash
python examples/run_mvp_linear.py
python examples/plot_linear.py
python examples/run_mvp_bifurcation.py
python examples/plot_bifurcation.py
```

## Output Naming

Current generated output groups are:

- `outputs/linear_20gene/`: small linear MVP sanity dataset.
- `outputs/bifurcation_20gene_3master/`: small hand-built GRN bifurcation dataset.

## Notes on SERGIO-derived Workflows

SERGIO-derived example scripts are intentionally not part of the public GitHub example surface anymore because they depend on external datasets or local-only workflows.

Those scripts are still kept on the remote research environment for ad hoc use, but they are not part of the fresh-clone reproducibility contract of this repository.
