# NVSim

Lightweight GRN-aware RNA velocity simulator.

## Quick-look plotting notes

NVSim plotting utilities in `nvsim.plotting` are intended for qualitative inspection, not publication-ready velocity inference.

- Layer matrices use the convention `cells x genes`.
- Phase portraits default to `mode="true"`, using `true_unspliced` and `true_spliced`, because observed UMI counts can become highly discrete after capture and Poisson noise. Use `mode="observed"` when you specifically want to inspect the noisy observed counts.
- PCA on `true_spliced` is the default view for checking smooth simulated dynamics.
- PCA or UMAP on observed `spliced` is useful for checking what downstream methods would see, but UMAP from sparse/noisy observed counts can fragment simple trajectories.
- Velocity arrows in `plot_embedding_with_velocity` default to PCA. They are PCA-projected quick-look diagnostics, not full scVelo-style velocity embedding. If `method="umap"` is explicitly requested, the arrows remain PCA-projected and are only overlaid on UMAP coordinates for qualitative visualization.

Run:

```bash
python examples/plot_linear.py
```

This writes PNG files under `examples/plots_linear/`:

- `true/`: primary scientific diagnostics. Start with `embedding_pca_true_by_pseudotime.png` to assess continuous underlying dynamics, then `embedding_pca_true_with_velocity.png` to check whether projected true velocity follows the trajectory. Use true phase portraits and gene dynamics to inspect GRN-regulated kinetics.
- `observed/`: noisy observed-count diagnostics. Use these to inspect technical noise effects, not to validate the smooth ODE directly.
- `observed_lownoise/`: observed diagnostics under a mild noise setting (`capture_rate=1.0`, `dropout_rate=0.0`) to separate count sampling effects from the underlying dynamics.
- `diagnostics/`: notes or future summary files.

Observed UMAP can fragment sparse toy datasets and should not be overinterpreted. PCA velocity arrows are qualitative diagnostics, not a full scVelo-style velocity embedding.

## Bifurcation MVP

`simulate_bifurcation` implements the current trunk-to-two-branch MVP. NVSim
integrates the trunk first, copies the terminal trunk `u` and `s` vectors at the
branch point, then integrates `branch_0` and `branch_1` independently from that
inherited state. Returned layer matrices are cells x genes; internal segment
arrays stored in `uns["segment_time_courses"]` are timepoints x genes. Cell order
is trunk samples first, then `branch_0`, then `branch_1`; `obs` includes global
`pseudotime`, segment-local `local_time`, and `branch` labels.

Branch-specific master regulator programs can be supplied through
`branch_master_programs`. If they are omitted, both branches share the same
programs and can remain dynamically identical after inheriting the same state.
This MVP still does not implement gene classes, promoter switching, SERGIO CLE,
or VeloSim EVF-to-kinetics.

Run `python examples/run_mvp_bifurcation.py` to generate a toy bifurcation h5ad
when AnnData is installed. Run `python examples/plot_bifurcation.py` to create
quick-look plots under `examples/plots_bifurcation/`. Start with true PCA by
branch and true PCA with velocity arrows to inspect the underlying bifurcation.
Use observed and low-noise observed plots only to inspect technical noise effects;
observed UMAP can fragment sparse toy data and should not be overinterpreted.
PCA velocity arrows are qualitative diagnostics, not a full scVelo-style velocity
embedding.

## Plotting Refinements

True-layer plots remain the primary scientific validation view for NVSim.
Observed plots are technical-noise diagnostics. The `observed_lownoise` example
outputs use `capture_rate=1.0`, `dropout_rate=0.0`, and
`poisson_observed=False`, so they are continuous visualization/debugging views
rather than realistic UMI count simulations. Bifurcation plotting selects
representative master, activation-target, and repression-target genes by
post-branch `true_alpha` divergence when branch labels are available.
