#!/usr/bin/env python3
"""Prepare a TRRUST v2 mouse signed GRN benchmark dataset for NVSim."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nvsim.grn import GRN
from nvsim.production import StateProductionProfile

TRRUST_MOUSE_URL = "https://www.grnpedia.org/trrust/data/trrust_rawdata.mouse.tsv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "external" / "trrust_mouse"
STANDARD_COLUMNS = [
    "regulator",
    "target",
    "sign",
    "K",
    "half_response",
    "hill_coefficient",
    "source",
    "mode",
    "pmids",
]
SIGNED_MODES = {"activation": "Activation", "repression": "Repression", "unknown": "Unknown"}


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _normalize_mode(value: str) -> str:
    key = str(value).strip().lower()
    if key not in SIGNED_MODES:
        raise ValueError(f"unrecognized TRRUST mode {value!r}")
    return SIGNED_MODES[key]


def _looks_like_header(row: list[str]) -> bool:
    if len(row) < 3:
        return False
    head0 = row[0].strip().lower()
    head1 = row[1].strip().lower()
    head2 = row[2].strip().lower()
    return head0 in {"tf", "transcription factor"} and head1.startswith("target") and "mode" in head2


def read_trrust_tsv(path: str | Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with Path(path).open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if row and any(field.strip() for field in row):
                rows.append([field.strip() for field in row])
    if not rows:
        raise ValueError(f"TRRUST TSV is empty: {path}")
    return rows


def download_trrust_mouse_tsv(output_path: str | Path = DEFAULT_OUTPUT_DIR / "trrust_rawdata.mouse.tsv") -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(TRRUST_MOUSE_URL, output_path)
    return output_path


def build_signed_full_grn(rows: list[list[str]]) -> tuple[pd.DataFrame, dict[str, int]]:
    body = rows[1:] if _looks_like_header(rows[0]) else rows
    pair_modes: dict[tuple[str, str], set[str]] = defaultdict(set)
    pair_pmids: dict[tuple[str, str], list[str]] = defaultdict(list)
    stats = {
        "raw_edges": 0,
        "signed_edges": 0,
        "dropped_unknown_edges": 0,
        "dropped_conflicting_pairs": 0,
        "dropped_self_loops": 0,
    }

    for row_number, row in enumerate(body, start=1):
        if len(row) < 4:
            raise ValueError(f"TRRUST row {row_number} must have at least 4 tab-separated columns")
        tf, target, mode_raw, pmid_field = row[:4]
        mode = _normalize_mode(mode_raw)
        stats["raw_edges"] += 1
        if mode == "Unknown":
            stats["dropped_unknown_edges"] += 1
        else:
            stats["signed_edges"] += 1
        pair = (tf, target)
        pair_modes[pair].add(mode)
        pair_pmids[pair].extend(str(token).strip() for token in pmid_field.split(";") if str(token).strip())

    records: list[dict[str, object]] = []
    for (tf, target), modes in sorted(pair_modes.items()):
        if tf == target:
            stats["dropped_self_loops"] += 1
            continue
        signed = {mode for mode in modes if mode in {"Activation", "Repression"}}
        if not signed:
            continue
        if len(signed) > 1:
            stats["dropped_conflicting_pairs"] += 1
            continue
        mode = next(iter(signed))
        records.append(
            {
                "regulator": tf,
                "target": target,
                "sign": "+" if mode == "Activation" else "-",
                "K": 1.0,
                "half_response": 1.0,
                "hill_coefficient": 2.0,
                "source": "TRRUST_v2_mouse",
                "mode": mode,
                "pmids": ";".join(_dedupe_preserve_order(pair_pmids[(tf, target)])),
            }
        )

    if not records:
        raise ValueError("no signed TRRUST edges remained after filtering")

    full_df = pd.DataFrame.from_records(records, columns=STANDARD_COLUMNS)
    full_df = full_df.sort_values(["regulator", "target", "mode"]).reset_index(drop=True)
    return full_df, stats


def _sorted_genes(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values})


def _tf_order(out_degree: Counter[str]) -> list[str]:
    return sorted(out_degree, key=lambda gene: (-out_degree[gene], gene))


def build_small_benchmark_grn(
    full_df: pd.DataFrame,
    *,
    top_tfs: int = 60,
    max_genes: int = 500,
    max_edges: int = 1500,
    max_in_degree: int = 5,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if top_tfs <= 0 or max_genes <= 0 or max_edges <= 0 or max_in_degree <= 0:
        raise ValueError("top_tfs, max_genes, max_edges, and max_in_degree must be positive")
    if top_tfs > max_genes:
        raise ValueError("top_tfs cannot exceed max_genes because selected TFs are always retained")

    out_degree_full: Counter[str] = Counter(full_df["regulator"].astype(str))
    top_tf_list = _tf_order(out_degree_full)[: min(top_tfs, len(out_degree_full))]

    candidate = full_df.loc[full_df["regulator"].astype(str).isin(top_tf_list)].copy()
    candidate["_reg_out_degree"] = candidate["regulator"].map(out_degree_full).astype(int)
    candidate = candidate.sort_values(["target", "_reg_out_degree", "regulator", "sign"], ascending=[True, False, True, True])
    candidate = candidate.groupby("target", sort=False, group_keys=False).head(max_in_degree).copy()

    selected_genes = set(top_tf_list)
    kept_targets = {target for target in candidate["target"].astype(str) if target in selected_genes}
    if len(selected_genes | set(candidate["target"].astype(str))) > max_genes:
        target_in_degree = candidate.groupby("target").size()
        target_reg_score = candidate.groupby("target")["_reg_out_degree"].max()
        ordered_targets = sorted(
            target_in_degree.index.astype(str),
            key=lambda target: (-int(target_in_degree[target]), -int(target_reg_score[target]), str(target)),
        )
        for target in ordered_targets:
            if target in kept_targets:
                continue
            maybe_genes = selected_genes | {target}
            if len(maybe_genes) <= max_genes:
                kept_targets.add(target)
                selected_genes = maybe_genes
        candidate = candidate.loc[candidate["target"].astype(str).isin(kept_targets)].copy()

    if len(candidate) > max_edges:
        target_in_degree = candidate.groupby("target").size()
        candidate = candidate.assign(_target_in_degree=candidate["target"].map(target_in_degree).astype(int))
        candidate = candidate.sort_values(
            ["_reg_out_degree", "_target_in_degree", "target", "regulator", "sign"],
            ascending=[False, False, True, True, True],
        ).head(max_edges).copy()
    else:
        candidate = candidate.assign(_target_in_degree=candidate.groupby("target")["target"].transform("size").astype(int))

    small_df = candidate.loc[:, STANDARD_COLUMNS].copy().reset_index(drop=True)
    metadata = {
        "top_tfs_selected": list(top_tf_list),
        "top_tf_out_degrees": {tf: int(out_degree_full[tf]) for tf in top_tf_list},
        "max_in_degree": int(max_in_degree),
        "max_genes": int(max_genes),
        "max_edges": int(max_edges),
        "small_grn_max_in_degree": int(candidate["_target_in_degree"].max()) if not candidate.empty else 0,
    }
    return small_df, metadata


def choose_master_regulators(small_df: pd.DataFrame, *, n_masters: int = 10, min_zero_indegree: int = 5) -> list[str]:
    if small_df.empty:
        raise ValueError("small_df must not be empty")
    tfs = _sorted_genes(small_df["regulator"].astype(str))
    target_in_degree: Counter[str] = Counter(small_df["target"].astype(str))
    tf_out_degree: Counter[str] = Counter(small_df["regulator"].astype(str))
    zero_indegree_tfs = sorted([tf for tf in tfs if target_in_degree[tf] == 0], key=lambda tf: (-tf_out_degree[tf], tf))

    selected: list[str] = []
    selected.extend(zero_indegree_tfs[: min(n_masters, len(zero_indegree_tfs))])
    if len(zero_indegree_tfs) < min_zero_indegree:
        for tf in sorted(tfs, key=lambda tf: (-tf_out_degree[tf], tf)):
            if tf in selected:
                continue
            selected.append(tf)
            if len(selected) >= min(n_masters, len(tfs)):
                break
    else:
        for tf in zero_indegree_tfs[len(selected):]:
            if len(selected) >= min(n_masters, len(tfs)):
                break
            selected.append(tf)
    if len(selected) < min(n_masters, len(tfs)):
        for tf in sorted(tfs, key=lambda tf: (-tf_out_degree[tf], tf)):
            if tf in selected:
                continue
            selected.append(tf)
            if len(selected) >= min(n_masters, len(tfs)):
                break
    return selected[: min(n_masters, len(tfs))]


def make_master_production_profile(master_regulators: list[str], *, seed: int = 1) -> pd.DataFrame:
    if not master_regulators:
        raise ValueError("at least one master regulator is required")
    rng = np.random.default_rng(seed)
    baseline = rng.uniform(0.5, 2.0, size=len(master_regulators))
    direction = rng.choice(np.array([-1.0, 1.0]), size=len(master_regulators))
    delta = rng.uniform(0.15, 0.6, size=len(master_regulators))
    root = baseline
    branch0 = np.clip(baseline + direction * delta, 0.3, 3.0)
    branch1 = np.clip(baseline - direction * delta, 0.3, 3.0)
    for idx in range(len(master_regulators)):
        if np.isclose(branch0[idx], branch1[idx]):
            branch1[idx] = np.clip(root[idx] - direction[idx] * max(delta[idx], 0.2), 0.3, 3.0)
    profile = pd.DataFrame(
        [root, branch0, branch1],
        index=pd.Index(["root", "branch0", "branch1"], name="state"),
        columns=[str(gene) for gene in master_regulators],
        dtype=float,
    )
    return profile.round(6)


def validate_outputs(full_df: pd.DataFrame, small_df: pd.DataFrame, masters: list[str], production_profile: pd.DataFrame) -> None:
    full_genes = _sorted_genes(pd.concat([full_df["regulator"], full_df["target"]], axis=0))
    small_genes = _sorted_genes(pd.concat([small_df["regulator"], small_df["target"]], axis=0))
    GRN.from_dataframe(full_df.loc[:, ["regulator", "target", "sign", "K", "half_response", "hill_coefficient"]], genes=full_genes)
    GRN.from_dataframe(
        small_df.loc[:, ["regulator", "target", "sign", "K", "half_response", "hill_coefficient"]],
        genes=small_genes,
        master_regulators=masters,
    )
    StateProductionProfile(production_profile)


def write_outputs(
    output_dir: str | Path,
    full_df: pd.DataFrame,
    small_df: pd.DataFrame,
    master_regulators: list[str],
    production_profile: pd.DataFrame,
    summary: dict[str, object],
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / "trrust_mouse_signed_full.csv"
    small_path = output_dir / "trrust_mouse_small_grn.csv"
    masters_path = output_dir / "trrust_mouse_master_regulators.txt"
    production_path = output_dir / "trrust_mouse_mr_production.csv"
    summary_path = output_dir / "summary.json"

    full_df.to_csv(full_path, index=False)
    small_df.to_csv(small_path, index=False)
    masters_path.write_text("\n".join(master_regulators) + "\n", encoding="utf-8")
    production_profile.reset_index().to_csv(production_path, index=False)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return {
        "full_grn_csv": str(full_path),
        "small_grn_csv": str(small_path),
        "master_regulators_txt": str(masters_path),
        "mr_production_csv": str(production_path),
        "summary_json": str(summary_path),
    }


def prepare_trrust_mouse_dataset(
    *,
    trrust_tsv: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    top_tfs: int = 60,
    max_genes: int = 500,
    max_edges: int = 1500,
    max_in_degree: int = 5,
    seed: int = 1,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    if trrust_tsv is None:
        raw_path = download_trrust_mouse_tsv(output_dir / "trrust_rawdata.mouse.tsv")
    else:
        raw_path = Path(trrust_tsv)
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)

    rows = read_trrust_tsv(raw_path)
    full_df, stats = build_signed_full_grn(rows)
    small_df, small_metadata = build_small_benchmark_grn(
        full_df,
        top_tfs=top_tfs,
        max_genes=max_genes,
        max_edges=max_edges,
        max_in_degree=max_in_degree,
    )
    masters = choose_master_regulators(small_df, n_masters=10, min_zero_indegree=5)
    production_profile = make_master_production_profile(masters, seed=seed)
    validate_outputs(full_df, small_df, masters, production_profile)

    full_genes = _sorted_genes(pd.concat([full_df["regulator"], full_df["target"]], axis=0))
    full_tfs = _sorted_genes(full_df["regulator"])
    small_genes = _sorted_genes(pd.concat([small_df["regulator"], small_df["target"]], axis=0))
    small_tfs = _sorted_genes(small_df["regulator"])

    summary = {
        "raw_source": str(raw_path),
        "raw_edges": int(stats["raw_edges"]),
        "signed_edges": int(stats["signed_edges"]),
        "dropped_unknown_edges": int(stats["dropped_unknown_edges"]),
        "dropped_conflicting_pairs": int(stats["dropped_conflicting_pairs"]),
        "dropped_self_loops": int(stats["dropped_self_loops"]),
        "full_grn_genes": int(len(full_genes)),
        "full_grn_tfs": int(len(full_tfs)),
        "full_grn_edges": int(full_df.shape[0]),
        "small_grn_genes": int(len(small_genes)),
        "small_grn_tfs": int(len(small_tfs)),
        "small_grn_edges": int(small_df.shape[0]),
        "activation_edges": int((small_df["mode"] == "Activation").sum()),
        "repression_edges": int((small_df["mode"] == "Repression").sum()),
        "master_regulators": masters,
        "parameters": {
            "top_tfs": int(top_tfs),
            "max_genes": int(max_genes),
            "max_edges": int(max_edges),
            "max_in_degree": int(max_in_degree),
            "seed": int(seed),
        },
        "subnetwork": small_metadata,
    }
    output_files = write_outputs(output_dir, full_df, small_df, masters, production_profile, {**summary, "output_files": {}})
    summary["output_files"] = output_files
    with (Path(output_dir) / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trrust_tsv", type=Path, default=None, help="Optional local TRRUST mouse TSV path")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top_tfs", type=int, default=60)
    parser.add_argument("--max_genes", type=int, default=500)
    parser.add_argument("--max_edges", type=int, default=1500)
    parser.add_argument("--max_in_degree", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = prepare_trrust_mouse_dataset(
        trrust_tsv=args.trrust_tsv,
        output_dir=args.output_dir,
        top_tfs=args.top_tfs,
        max_genes=args.max_genes,
        max_edges=args.max_edges,
        max_in_degree=args.max_in_degree,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
