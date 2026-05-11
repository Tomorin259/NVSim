from pathlib import Path

import pandas as pd
import pytest

from nvsim.sergio_io import load_sergio_targets_regs


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_load_sergio_targets_regs_maps_edges_and_master_production(tmp_path):
    targets = _write(
        tmp_path / "targets.csv",
        "\n".join(
            [
                "1,1,0,2.5,2",
                "2,2,0,1,-1.0,0.7,3,1.5",
            ]
        ),
    )
    regs = _write(tmp_path / "regs.csv", "0,0.5,1.0,1.5")

    parsed = load_sergio_targets_regs(targets, regs)
    edges = parsed.grn.to_dataframe()

    assert parsed.grn.genes == ("0", "1", "2")
    assert parsed.master_regulators == ("0",)
    assert parsed.grn.master_regulators == ("0",)
    assert parsed.master_production.equals(
        pd.DataFrame(
            {"0": [0.5, 1.0, 1.5]},
            index=pd.Index(["bin_0", "bin_1", "bin_2"], name="state"),
            dtype=float,
        ).rename_axis(columns="gene")
    )
    assert list(edges["regulator"]) == ["0", "0", "1"]
    assert list(edges["target"]) == ["1", "2", "2"]
    assert list(edges["K"]) == [2.5, 1.0, 0.7]
    assert list(edges["sign"]) == ["activation", "repression", "activation"]
    assert list(edges["hill_coefficient"]) == [2.0, 3.0, 1.5]


def test_load_sergio_targets_regs_supports_shared_coop_state(tmp_path):
    targets = _write(tmp_path / "targets.csv", "1,1,0,-1.2")
    regs = _write(tmp_path / "regs.csv", "0,0.5,1.0")

    parsed = load_sergio_targets_regs(targets, regs, shared_coop_state=2.5)
    edges = parsed.grn.to_dataframe()

    assert list(edges["sign"]) == ["repression"]
    assert list(edges["K"]) == [1.2]
    assert list(edges["hill_coefficient"]) == [2.5]


def test_load_sergio_targets_regs_rejects_master_as_target(tmp_path):
    targets = _write(tmp_path / "targets.csv", "0,1,1,2.0,2")
    regs = _write(tmp_path / "regs.csv", "0,0.5\n1,1.0")

    with pytest.raises(ValueError, match="must not appear as targets"):
        load_sergio_targets_regs(targets, regs)


def test_load_sergio_targets_regs_rejects_unknown_regulator(tmp_path):
    targets = _write(tmp_path / "targets.csv", "2,1,1,2.0,2")
    regs = _write(tmp_path / "regs.csv", "0,0.5")

    with pytest.raises(ValueError, match="regulators missing"):
        load_sergio_targets_regs(targets, regs)


def test_load_sergio_targets_regs_rejects_inconsistent_bins(tmp_path):
    targets = _write(tmp_path / "targets.csv", "1,1,0,2.0,2")
    regs = _write(tmp_path / "regs.csv", "0,0.5,1.0\n2,0.5")

    with pytest.raises(ValueError, match="same number"):
        load_sergio_targets_regs(targets, regs)
