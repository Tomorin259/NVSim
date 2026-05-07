"""SERGIO-style GRN input parsing.

This module reads SERGIO ``targets`` and ``regs`` CSV files and converts them
into NVSim inputs without using SERGIO's stochastic simulator. The parser keeps
SERGIO's GRN-to-production-rate semantics:

``targets`` rows define target genes, regulators, signed interaction strengths
``K``, and Hill/cooperativity coefficients. ``regs`` rows define master
regulator production rates for each bin/state.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..grn import GRN


@dataclass(frozen=True)
class SergioInputs:
    """Parsed SERGIO-style inputs for NVSim.

    ``grn`` is an NVSim GRN with non-negative edge weights and normalized edge
    signs. ``master_production`` is a bins x master-regulators matrix whose
    columns are gene ids and whose rows are ``bin_0``, ``bin_1``, ...
    """

    grn: GRN
    master_production: pd.DataFrame
    master_regulators: tuple[str, ...]


def _read_csv_rows(path: str | Path) -> list[list[str]]:
    with Path(path).open(newline="") as handle:
        return [
            [field.strip() for field in row]
            for row in csv.reader(handle)
            if row and any(field.strip() for field in row)
        ]


def _as_gene_id(value: str) -> str:
    numeric = float(value)
    if not numeric.is_integer():
        raise ValueError(f"gene ids must be integer-like in SERGIO inputs, got {value!r}")
    return str(int(numeric))


def _as_positive_int(value: str, name: str) -> int:
    numeric = float(value)
    if not numeric.is_integer() or numeric <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(numeric)


def _gene_sort_key(gene: str) -> tuple[int, int | str]:
    try:
        return (0, int(gene))
    except ValueError:
        return (1, gene)


def _ordered_genes(*groups: Iterable[str]) -> list[str]:
    genes: set[str] = set()
    for group in groups:
        genes.update(str(gene) for gene in group)
    return sorted(genes, key=_gene_sort_key)


def _parse_targets_rows(rows: list[list[str]], shared_coop_state: float) -> tuple[pd.DataFrame, set[str], set[str]]:
    records = []
    regulators: set[str] = set()
    targets: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        if len(row) < 2:
            raise ValueError(f"targets row {row_number} must contain at least target id and regulator count")

        target = _as_gene_id(row[0])
        n_regs = _as_positive_int(row[1], f"targets row {row_number} regulator count")
        required_without_coop = 2 + 2 * n_regs
        required_with_coop = 2 + 3 * n_regs
        if shared_coop_state <= 0 and len(row) < required_with_coop:
            raise ValueError(f"targets row {row_number} is missing Hill/cooperativity values")
        if shared_coop_state > 0 and len(row) < required_without_coop:
            raise ValueError(f"targets row {row_number} is missing regulator or K values")

        reg_values = row[2 : 2 + n_regs]
        k_values = row[2 + n_regs : 2 + 2 * n_regs]
        coop_values = (
            [str(shared_coop_state)] * n_regs
            if shared_coop_state > 0
            else row[2 + 2 * n_regs : 2 + 3 * n_regs]
        )

        targets.add(target)
        for regulator_value, k_value, coop_value in zip(reg_values, k_values, coop_values):
            regulator = _as_gene_id(regulator_value)
            k = float(k_value)
            hill = float(coop_value)
            if k == 0:
                raise ValueError(f"targets row {row_number} has zero K for regulator {regulator}")
            if hill <= 0:
                raise ValueError(f"targets row {row_number} has non-positive Hill coefficient {hill}")
            regulators.add(regulator)
            records.append(
                {
                    "regulator": regulator,
                    "target": target,
                    "K": abs(k),
                    "sign": "activation" if k > 0 else "repression",
                    "hill_coefficient": hill,
                }
            )

    if not records:
        raise ValueError("targets file did not contain any regulatory edges")
    return pd.DataFrame.from_records(records), regulators, targets


def _parse_regs_rows(rows: list[list[str]]) -> tuple[pd.DataFrame, list[str]]:
    if not rows:
        raise ValueError("regs file did not contain any master regulators")

    master_ids: list[str] = []
    production_columns: list[list[float]] = []
    n_bins: int | None = None
    for row_number, row in enumerate(rows, start=1):
        if len(row) < 2:
            raise ValueError(f"regs row {row_number} must contain a master id and at least one production rate")
        master_id = _as_gene_id(row[0])
        rates = [float(value) for value in row[1:]]
        if any(rate < 0 for rate in rates):
            raise ValueError(f"regs row {row_number} contains a negative production rate")
        if n_bins is None:
            n_bins = len(rates)
        elif len(rates) != n_bins:
            raise ValueError("all regs rows must contain the same number of bins/states")
        master_ids.append(master_id)
        production_columns.append(rates)

    if len(set(master_ids)) != len(master_ids):
        raise ValueError("regs file contains duplicate master regulator ids")

    production = pd.DataFrame(
        {master_id: rates for master_id, rates in zip(master_ids, production_columns)},
        index=[f"bin_{idx}" for idx in range(n_bins or 0)],
        dtype=float,
    )
    production.index.name = "state"
    production.columns.name = "gene"
    return production, master_ids


def load_sergio_targets_regs(
    targets_file: str | Path,
    regs_file: str | Path,
    shared_coop_state: float = 0,
) -> SergioInputs:
    """Load SERGIO ``targets`` and ``regs`` files into NVSim inputs.

    Parameters
    ----------
    targets_file:
        CSV file with rows ``target, n_regs, regs..., K..., hill...``. If
        ``shared_coop_state > 0``, per-edge Hill values are not required and the
        shared value is used for every edge.
    regs_file:
        CSV file with rows ``master, production_rate_bin_0, ...``.
    shared_coop_state:
        SERGIO-compatible override for all Hill/cooperativity coefficients.

    Returns
    -------
    SergioInputs
        A dataclass containing ``grn`` and ``master_production``.
    """

    if shared_coop_state < 0:
        raise ValueError("shared_coop_state must be non-negative")

    target_edges, regulators, targets = _parse_targets_rows(_read_csv_rows(targets_file), shared_coop_state)
    master_production, master_ids = _parse_regs_rows(_read_csv_rows(regs_file))

    master_set = set(master_ids)
    if master_set & targets:
        overlap = sorted(master_set & targets, key=_gene_sort_key)
        raise ValueError(f"master regulators must not appear as targets in SERGIO inputs: {overlap}")
    if not regulators.issubset(master_set | targets):
        unknown = sorted(regulators - master_set - targets, key=_gene_sort_key)
        raise ValueError(f"regulators missing from master regs or target rows: {unknown}")

    genes = _ordered_genes(master_ids, targets)
    return SergioInputs(
        grn=GRN.from_dataframe(target_edges, genes=genes, master_regulators=master_ids),
        master_production=master_production,
        master_regulators=tuple(master_ids),
    )
