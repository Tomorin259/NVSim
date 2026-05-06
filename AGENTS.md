# NVSim project instructions

## Goal

This project builds NVSim, a lightweight GRN-aware RNA velocity simulator.

The modeling chain is:

GRN -> transcription rate alpha(t) -> unspliced RNA u(t) -> spliced RNA s(t) -> true RNA velocity v(t)

NVSim should combine ideas from:

- SERGIO: GRN-guided transcriptional regulation, regulator-target edges, activation/repression, regulatory weights, Hill-type response, and optional technical-noise ideas.
- VeloSim: trajectory-aware simulation, pseudotime, branch/lineage assignment, unspliced/spliced kinetic simulation, snapshot sampling, and true RNA velocity output.

NVSim should NOT be a dyngen-style simulator. Do not implement molecule-level SSA, protein/translation layers, or heavy reaction bookkeeping.

## Core scientific model

For the MVP, the GRN controls transcription rate alpha_i(t).

The u/s dynamics should be:

du_i/dt = alpha_i(t) - beta_i * u_i(t)

ds_i/dt = beta_i * u_i(t) - gamma_i * s_i(t)

true spliced RNA velocity:

v_i(t) = beta_i * u_i(t) - gamma_i * s_i(t)

For the first version:

- alpha_i(t) is cell/time-specific and GRN-controlled.
- beta_i and gamma_i are gene-specific vectors.
- Later versions may allow beta_i(t) and gamma_i(t) to be cell-specific or branch-specific.

## Engineering rules

- Do not directly modify the original SERGIO or VeloSim folders.
- Treat SERGIO and VeloSim as read-only references.
- Implement NVSim inside simulator/NVSim.
- Prefer Python for the new implementation.
- Core simulation code should depend only on numpy and pandas where possible.
- scipy can be used for ODE integration if available.
- anndata should be optional and only used for h5ad export.
- No network calls.
- Do not install packages unless explicitly asked.
- If a dependency is missing, report it clearly and add it to requirements instead of silently changing the environment.
- Keep functions small, testable, and documented.
- Use deterministic random seeds.
- Add sanity checks or minimal tests.

## Expected AnnData output

The simulator should eventually produce an AnnData object with:

- adata.layers["unspliced"]
- adata.layers["spliced"]
- adata.layers["true_unspliced"]
- adata.layers["true_spliced"]
- adata.layers["true_velocity"]
- adata.obs["pseudotime"]
- adata.obs["branch"]
- adata.obs["cell_type"] if available
- adata.var["gene_class"] if available
- adata.uns["true_grn"]
- adata.uns["kinetic_params"]
- adata.uns["simulation_config"]

## Scientific constraints

NVSim is designed for benchmarking RNA velocity methods and gene-level kinetics.

Therefore, the following ground truth must be accessible and reproducible:

- GRN
- alpha
- beta
- gamma
- true u
- true s
- true velocity
- pseudotime
- branch labels
- gene classes