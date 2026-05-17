# TRRUST v2 Mouse Signed GRN for NVSim

This directory stores the first external GRN benchmark input derived from TRRUST v2 mouse.

## Source

- Raw source URL: https://www.grnpedia.org/trrust/data/trrust_rawdata.mouse.tsv
- Database: TRRUST v2 mouse TF-target regulatory relationships
- Raw fields: TF, Target, Mode, PMID

## Filtering Rules

The prepared NVSim dataset keeps only signed edges.

- Keep Mode == Activation or Mode == Repression
- Drop Mode == Unknown
- Drop TF-target pairs with conflicting signed labels
- Drop self-loops where TF == Target
- Preserve original mouse gene-symbol casing

## Output Files

- trrust_rawdata.mouse.tsv: raw downloaded TSV when the prepare script uses the official URL
- trrust_mouse_signed_full.csv: full signed NVSim-formatted GRN
- trrust_mouse_small_grn.csv: bounded benchmark subnetwork
- trrust_mouse_master_regulators.txt: chosen master regulators for the small benchmark
- trrust_mouse_mr_production.csv: simple state-wise master-regulator production profile
- summary.json: dataset statistics and output paths

## NVSim Edge Columns

Prepared CSVs use these columns:

- regulator
- target
- sign
- K
- half_response
- hill_coefficient
- source
- mode
- pmids

For the first version:

- Activation -> sign = +
- Repression -> sign = -
- K = 1.0
- half_response = 1.0
- hill_coefficient = 2

The example simulation then uses the NVSim runtime calibration path with
auto_calibrate_half_response=True so the benchmark is not locked to the
placeholder threshold value.

## How To Build

From the repository root:

python examples/prepare_trrust_mouse_dataset.py

Or with a local TRRUST TSV:

python examples/prepare_trrust_mouse_dataset.py --trrust_tsv /path/to/trrust_rawdata.mouse.tsv

## Example Simulation

python examples/run_trrust_mouse_small.py
