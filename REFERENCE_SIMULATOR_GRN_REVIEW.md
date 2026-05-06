# Reference Simulator GRN Review

This note records the read-only inspection of the simulator folders adjacent to NVSim.
No files under `../SERGIO`, `../VeloSim`, `../dyngen`, or `../scMultiSim` were modified.

## Summary

The most directly reusable GRN sources are in SERGIO. The other simulators are useful for design ideas but do not expose static GRN files as directly:

- SERGIO: includes explicit GRN input files and GNW DOT networks with activation/repression signs.
- dyngen: tutorials generate large TF/module networks programmatically; useful conceptually, but not a simple static GRN file for NVSim to read without running dyngen.
- scMultiSim: local run scripts use built-in `GRN_params_100` and dynamic GRN machinery; useful reference, less transparent than SERGIO DOT files for immediate reuse.
- VeloSim: focuses on trajectory/EVF/kinetic simulation rather than GRN input.

## SERGIO GRN Candidates

### GNW DOT Networks

- `Ecoli_100_net1.dot`: 100 genes, 137 edges, 6 master genes. Top master regulators include `arcA`, `glnG`, and `phoP`.
- `Ecoli_100_net2.dot`: 100 genes, 241 edges, 3 master genes. Top master regulators include `narP` and `modE`.
- `Yeast_400_net3.dot`: 400 genes, 1157 edges, 19 master genes. Top master regulators include `YIR018W`, `YDR146C`, `YNL216W`, `YPL248C`, and `YDR043C`.
- `Ecoli_1200_net4.dot`: 1200 genes, 2644 edges, 53 master genes. Top master regulators include `arcA`, `nsrR`, `modE`, `fruR`, `narP`, and `phoP`.

`Yeast_400_net3.dot` is the best immediate NVSim example: it is substantially larger than the toy 20-gene GRN but still quick enough for v0.1 examples. `Ecoli_1200_net4.dot` is a good next stress-test candidate when runtime and plotting are ready.

### SERGIO data_sets

SERGIO also includes denoised datasets with `gt_GRN.csv` files, including 100G, 400G, and 1200G configurations. These are useful references, but the two-column files require additional interpretation of sign/weight compared with the DOT files, so they were not used for the current NVSim conversion.

## dyngen Notes

The dyngen showcase uses examples like `backbone_bifurcating()`, `backbone_branching()`, and `backbone_binary_tree()` with `num_tfs = 100`. This supports the idea that a richer multi-master or multi-module driver set is needed to make manifolds wider and more realistic. However, these examples generate networks inside dyngen and are not simple static GRNs to import directly.

## scMultiSim Notes

The local full kinetic script uses:

- `grn_dataset = "GRN_params_100"`
- tree types such as `Phyla3`
- full kinetic RNA velocity outputs

This is useful as a reference for combining GRN-like effects, tree trajectories, and velocity outputs. For NVSim, the immediate reusable artifact is less direct than the SERGIO DOT networks.

## VeloSim Notes

VeloSim tutorials simulate 500 genes and 20 EVFs for tree trajectories, including unspliced, spliced, velocity, and technical noise. It is useful as a trajectory/velocity reference, but it is not a GRN source.

## Current NVSim Reuse

Added examples:

- `examples/run_sergio_grn_bifurcation.py`
- `examples/plot_sergio_grn_bifurcation.py`

These read `../SERGIO/GNW_sampled_GRNs/Yeast_400_net3.dot` as a read-only reference and convert it into NVSim's edge schema. Current output group:

- `examples/outputs/sergio_yeast400_3master/`

## Recommended Next Dataset

The next dataset should use the same `Yeast_400_net3.dot` GRN but drive more master regulators, for example top 8-12 master regulators by out-degree. This is a better next step than immediately jumping to `Ecoli_1200_net4.dot`, because it isolates whether the thin true manifold is caused by too few dynamic master programs rather than too few genes.
