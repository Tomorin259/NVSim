"""Run NVSim on a larger SERGIO/GNW example GRN.

This script reads a SERGIO-distributed GNW DOT network as a read-only GRN
reference and converts it to the NVSim edge schema. It does not modify SERGIO.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from nvsim.grn import GRN
from nvsim.output import to_anndata
from nvsim.production import linear_decrease, linear_increase, sigmoid_decrease, sigmoid_increase
from nvsim.simulate import simulate_bifurcation

DEFAULT_SERGIO_DOT = ROOT.parent / "SERGIO" / "GNW_sampled_GRNs" / "Yeast_400_net3.dot"


def load_sergio_dot_grn(path: str | Path = DEFAULT_SERGIO_DOT, seed: int = 2026) -> GRN:
    """Convert a SERGIO/GNW DOT GRN into NVSim's non-negative edge schema."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SERGIO DOT file not found: {path}")
    rng = np.random.default_rng(seed)
    nodes: set[str] = set()
    records = []
    edge_pattern = re.compile(r'"([^"]+)"\s*->\s*"([^"]+)"\s*\[value="([+-])"\]')
    node_pattern = re.compile(r'^\s*"([^"]+)"')
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        edge = edge_pattern.search(line)
        if edge:
            regulator, target, sign_symbol = edge.groups()
            nodes.update([regulator, target])
            records.append(
                {
                    "regulator": regulator,
                    "target": target,
                    "weight": float(rng.uniform(0.12, 0.45)),
                    "sign": "activation" if sign_symbol == "+" else "repression",
                    "threshold": float(rng.uniform(0.45, 0.9)),
                    "hill_coefficient": 2.0,
                }
            )
            continue
        node = node_pattern.search(line)
        if node and "->" not in line:
            nodes.add(node.group(1))
    if not records:
        raise ValueError(f"no DOT edges found in {path}")
    return GRN.from_dataframe(pd.DataFrame(records), genes=sorted(nodes))


def top_master_regulators(grn: GRN, n: int = 3) -> list[str]:
    incoming = set(grn.edges["target"].astype(str))
    out_degree = grn.edges.groupby("regulator").size().to_dict()
    masters = [gene for gene in grn.genes if gene not in incoming]
    ranked = sorted(masters, key=lambda gene: out_degree.get(gene, 0), reverse=True)
    return ranked[:n]


def build_sergio_grn_bifurcation_result(
    capture_rate: float = 0.6,
    dropout_rate: float = 0.01,
    poisson_observed: bool = True,
):
    """Generate a larger NVSim bifurcation dataset from the SERGIO Yeast-400 GRN."""

    grn = load_sergio_dot_grn()
    masters = top_master_regulators(grn, n=3)
    if len(masters) < 3:
        raise ValueError("expected at least three master regulators in the SERGIO GRN")
    master_programs = {
        masters[0]: linear_increase(0.25, 0.85),
        masters[1]: sigmoid_decrease(0.95, 0.30),
        masters[2]: 0.55,
    }
    branch_master_programs = {
        "branch_0": {
            masters[0]: linear_increase(0.85, 1.55),
            masters[1]: sigmoid_increase(0.30, 1.15, midpoint=0.65),
        },
        "branch_1": {
            masters[0]: linear_decrease(0.85, 0.20),
            masters[1]: sigmoid_decrease(0.30, 0.08, midpoint=0.65),
        },
    }
    result = simulate_bifurcation(
        grn,
        n_trunk_cells=180,
        n_branch_cells={"branch_0": 210, "branch_1": 210},
        trunk_time=2.0,
        branch_time=2.5,
        dt=0.05,
        master_programs=master_programs,
        branch_master_programs=branch_master_programs,
        alpha_max=4.0,
        seed=2026,
        capture_rate=capture_rate,
        poisson_observed=poisson_observed,
        dropout_rate=dropout_rate,
    )
    result["uns"]["simulation_config"]["source_grn"] = str(DEFAULT_SERGIO_DOT)
    result["uns"]["simulation_config"]["source_grn_n_genes"] = len(grn.genes)
    result["uns"]["simulation_config"]["source_grn_n_edges"] = int(grn.edges.shape[0])
    result["uns"]["simulation_config"]["branch_program_master_genes"] = masters
    return result


def main() -> None:
    result = build_sergio_grn_bifurcation_result()
    grn = result["uns"]["true_grn"]
    print(
        {
            "n_cells": result["layers"]["true_spliced"].shape[0],
            "n_genes": result["layers"]["true_spliced"].shape[1],
            "n_edges": int(grn.shape[0]),
            "branches": sorted(result["obs"]["branch"].unique()),
            "source_grn": result["uns"]["simulation_config"]["source_grn"],
            "branch_program_master_genes": result["uns"]["simulation_config"]["branch_program_master_genes"],
        }
    )
    try:
        adata = to_anndata(result)
    except ImportError:
        print("anndata is not installed; skipped h5ad export")
    else:
        out = Path(__file__).with_name("outputs") / "sergio_yeast400_3master" / "sergio_yeast400_bifurcation.h5ad"
        out.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(out)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
