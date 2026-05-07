# NVSim

NVSim is a lightweight GRN-aware RNA velocity simulator.

The current public checkpoint focuses on a transparent modeling chain:

```text
GRN -> alpha(t) -> unspliced/spliced ODE -> true velocity -> snapshot cells -> observed layers
```

It is intentionally smaller than SERGIO, VeloSim, dyngen, or scVelo-style pipelines. The goal of v0.1 is a clear benchmark scaffold with inspectable ground truth, not a full biological simulator.

## Overview

- Explicit GRN input with regulator, target, sign, `K`, `half_response`, and `hill_coefficient`.
- Explicit or inferred master regulators with state/bin-wise production profiles.
- SERGIO-style additive Hill-function production rates for non-master genes.
- Deterministic ODE dynamics for unspliced and spliced RNA.
- Separate true and observed layers.
- Linear and trunk-to-two-branch bifurcation examples.
- Optional AnnData export and quick-look plotting utilities.

## Install

Python 3.10 or newer is required.

Core install:

```bash
pip install -e .
```

Development install with tests and optional plotting dependencies:

```bash
pip install -e .[dev]
```

## Quick Start

Run the small linear example:

```bash
python examples/run_mvp_linear.py
python examples/plot_linear.py
```

Run the bifurcation example:

```bash
python examples/run_mvp_bifurcation.py
python examples/plot_bifurcation.py
```

Minimal Python usage:

```python
import pandas as pd

from nvsim.grn import GRN
from nvsim.production import StateProductionProfile
from nvsim.simulate import simulate_linear

edges = pd.DataFrame(
    {
        "regulator": ["g0"],
        "target": ["g1"],
        "K": [0.8],
        "sign": ["activation"],
        "half_response": [0.5],
        "hill_coefficient": [2.0],
    }
)

grn = GRN.from_dataframe(edges, genes=["g0", "g1"])
production = StateProductionProfile(
    pd.DataFrame({"g0": [1.0]}, index=["state_0"])
)
result = simulate_linear(
    grn,
    n_cells=50,
    time_end=2.0,
    dt=0.05,
    seed=7,
    master_regulators=["g0"],
    production_profile=production,
    production_state="state_0",
)

print(result["layers"]["true_spliced"].shape)
print(result["var"][["gene_role", "gene_class"]].head())
```

## Model Summary

For gene `i`, NVSim currently uses:

```text
du_i/dt = alpha_i(t) - beta_i * u_i(t)
ds_i/dt = beta_i * u_i(t) - gamma_i * s_i(t)
v_i(t) = beta_i * u_i(t) - gamma_i * s_i(t)
```

- `alpha_i(t)` is GRN-controlled.
- The core GRN layer follows a SERGIO-style parameterized production-rate model:
  - master regulators are explicit source genes;
  - master regulators use state/bin-wise production rates;
  - non-master genes receive additive Hill-function regulatory contributions.
- Canonical GRN parameters are `K`, `half_response`, and `hill_coefficient`.
- `gene_role` is `master_regulator` or `non_master`.
- `gene_class` is reserved for future biological class labels and is currently set to `unassigned`.
- Regulation uses additive Hill-style activation and repression contributions.
- State/bin-wise production is the default SERGIO-like forcing behavior.
- Continuous interpolation between production states is optional and remains an NVSim extension.
- Acyclic GRNs can use a SERGIO-style level-wise half-response calibration pass; cyclic GRNs fall back to a user/reference scale and are still supported by ODE time stepping.
- Observed counts can use either the original scale-plus-Poisson path or a VeloSim-style binomial capture model.
- Supported observed-count modes are `scale_poisson` and `binomial_capture`.
- `grn_calibration` and `noise_config` are stored in the plain result dict and carried into AnnData metadata.

## Current v0.1 Scope

Included now:

- Linear simulation.
- Trunk-to-two-branch bifurcation simulation.
- True/observed layer separation.
- Optional AnnData export.
- Qualitative PCA/UMAP plotting utilities.

Explicitly not included yet:

- Normal/MURK/branching gene classes.
- Promoter switching.
- SERGIO CLE or other stochastic simulator paths.
- Half-response auto-calibration as a default workflow. Calibration is available, but it is still an explicit preprocessing step rather than the default simulation path.
- VeloSim EVF-to-kinetics mapping.
- Full scVelo-style velocity embedding.
- Calibrated large-scale UMI realism.

## Project Docs

- [Current Status](CURRENT_STATUS.md)
- [Validation Report](VALIDATION_REPORT.md)
- [Chinese Model Notes](NVSim_model_cn.md)
- [Examples Guide](examples/README.md)
