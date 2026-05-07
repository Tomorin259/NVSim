# NVSim Examples

This directory contains reproducible scripts for the v0.1 MVP. Generated files are written under `outputs/` and are intentionally ignored by `.gitignore` because they can be regenerated from these scripts.

## Data Generation

- `run_mvp_linear.py`: builds a 20-gene linear GRN-aware RNA velocity dataset.
  If AnnData is installed, it writes `outputs/linear_20gene/mvp_linear.h5ad`.
- `run_mvp_bifurcation.py`: builds a trunk-to-two-branch dataset with inherited
  branch initial states. If AnnData is installed, it writes
  `outputs/bifurcation_20gene_3master/mvp_bifurcation.h5ad`.
- `run_sergio_grn_bifurcation.py`: reads the read-only SERGIO/GNW Yeast-400
  DOT GRN and builds a larger 400-gene NVSim bifurcation dataset. If AnnData is
  installed, it writes `outputs/sergio_yeast400_3master/sergio_yeast400_bifurcation.h5ad`.
- `run_sergio_grn_multimaster.py`: uses the same read-only Yeast-400 GRN but drives
  branch programs through ten high-out-degree master regulators. If AnnData is
  installed, it writes `outputs/sergio_yeast400_multimaster/sergio_yeast400_multimaster_bifurcation.h5ad`.
- `run_sergio_1200g_ds3_bifurcation.py`: the current recommended SERGIO-style
  deterministic path. It reads the original SERGIO dataset files
  `Interaction_cID_6.txt` and `Regs_cID_6.txt`, calibrates half-response values
  from SERGIO `simulated_noNoise_*.csv` mean expression, and writes results
  under `outputs/sergio_1200g_ds3_bifurcation/`.
- `run_sergio_100g_ds8_bifurcation.py`: reads SERGIO's original 100-gene
  3-state dynamics dataset files `Interaction_cID_8.txt`, `Regs_cID_8.txt`
  and `bMat_cID8.tab`, calibrates half-response values from
  `simulated_noNoise_*.csv` mean expression, and writes results under
  `outputs/sergio_100g_ds8_bifurcation/`.
- `run_sergio_ecoli1200_25master.py`: reads the read-only SERGIO/GNW Ecoli-1200
  DOT GRN and drives branch programs through 25 high-out-degree master regulators.
  If AnnData is installed, it writes `outputs/sergio_ecoli1200_25master/sergio_ecoli1200_25master_bifurcation.h5ad`.
  This is a legacy GNW/DOT stress-test path rather than the main SERGIO
  `targets/regs` workflow.

## Plotting

- `plot_linear.py`: generates quick-look figures under `outputs/linear_20gene/plots/`.
- `plot_bifurcation.py`: generates quick-look figures under
  `outputs/bifurcation_20gene_3master/plots/` and writes `diagnostics/selected_genes.txt`.
- `plot_sergio_grn_bifurcation.py`: generates quick-look figures under
  `outputs/sergio_yeast400_3master/plots/` for the larger SERGIO-derived GRN.
- `plot_sergio_grn_multimaster.py`: generates quick-look figures under
  `outputs/sergio_yeast400_multimaster/plots/` for the ten-master SERGIO-derived GRN.
- `plot_sergio_1200g_ds3_bifurcation.py`: generates quick-look figures under
  `outputs/sergio_1200g_ds3_bifurcation/plots/` for the SERGIO original
  1200-gene `targets/regs` dataset routed through NVSim's deterministic ODE path.
- `plot_sergio_100g_ds8_bifurcation.py`: generates quick-look figures under
  `outputs/sergio_100g_ds8_bifurcation/plots/` for SERGIO's original
  100-gene 3-state dynamics dataset routed through NVSim's deterministic ODE path.
- `plot_sergio_ecoli1200_25master.py`: generates quick-look figures under
  `outputs/sergio_ecoli1200_25master/plots/` for the 25-master Ecoli-1200 GRN.

Plot directories are organized as:

- `true/`: primary scientific validation plots from true layers.
- `observed/`: noisy observed-layer diagnostics.
- `observed_lownoise/`: continuous visualization/debugging views with
  `poisson_observed=False`.
- `diagnostics/`: text notes and selected-gene diagnostics.

## Recommended Check Commands

```bash
python examples/run_mvp_linear.py
python examples/plot_linear.py
python examples/run_mvp_bifurcation.py
python examples/plot_bifurcation.py
python examples/run_sergio_grn_bifurcation.py
python examples/plot_sergio_grn_bifurcation.py
python examples/run_sergio_grn_multimaster.py
python examples/plot_sergio_grn_multimaster.py
python examples/run_sergio_1200g_ds3_bifurcation.py
python examples/plot_sergio_1200g_ds3_bifurcation.py
python examples/run_sergio_100g_ds8_bifurcation.py
python examples/plot_sergio_100g_ds8_bifurcation.py
python examples/run_sergio_ecoli1200_25master.py
python examples/plot_sergio_ecoli1200_25master.py
```


## Output Naming

Current generated output groups are:

- `outputs/linear_20gene/`: small linear MVP sanity dataset.
- `outputs/bifurcation_20gene_3master/`: small hand-built GRN bifurcation dataset.
- `outputs/sergio_yeast400_3master/`: larger SERGIO/GNW Yeast-400 GRN dataset currently driven by three branch-program master regulators.
- `outputs/sergio_yeast400_multimaster/`: the same Yeast-400 GRN driven by ten branch-program master regulators; use this to inspect whether higher-dimensional master dynamics produce richer bifurcation structure.
- `outputs/sergio_1200g_ds3_bifurcation/`: SERGIO original `targets/regs`
  1200-gene dataset routed through NVSim's deterministic ODE path, with
  mean-expression half-response calibration from SERGIO `simulated_noNoise`
  reference data.
- `outputs/sergio_100g_ds8_bifurcation/`: SERGIO original 100-gene 3-state
  dynamics dataset routed through NVSim's deterministic ODE path, with
  mean-expression half-response calibration from SERGIO `simulated_noNoise`
  reference data and `bMat_cID8.tab` recorded as trajectory reference metadata.
- `outputs/sergio_ecoli1200_25master/`: larger SERGIO/GNW Ecoli-1200 GRN driven by 25 branch-program master regulators for stress-testing larger GRN-aware velocity simulation.

The SERGIO-derived output uses the read-only reference file:
`../SERGIO/GNW_sampled_GRNs/Yeast_400_net3.dot`.

The recommended 1200-gene SERGIO deterministic path uses the read-only
reference files:
`../SERGIO/data_sets/De-noised_1200G_9T_300cPerT_6_DS3/Interaction_cID_6.txt`
and
`../SERGIO/data_sets/De-noised_1200G_9T_300cPerT_6_DS3/Regs_cID_6.txt`.

The 100-gene SERGIO deterministic path uses the read-only reference files:
`../SERGIO/data_sets/De-noised_100G_3T_300cPerT_dynamics_8_DS8/Interaction_cID_8.txt`,
`../SERGIO/data_sets/De-noised_100G_3T_300cPerT_dynamics_8_DS8/Regs_cID_8.txt`
and
`../SERGIO/data_sets/De-noised_100G_3T_300cPerT_dynamics_8_DS8/bMat_cID8.tab`.

The Ecoli-1200 output uses the read-only reference file:
`../SERGIO/GNW_sampled_GRNs/Ecoli_1200_net4.dot`.
