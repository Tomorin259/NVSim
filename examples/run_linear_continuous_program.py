"""Run the canonical linear example for alpha_source_mode="continuous_program"."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nvsim.output import to_anndata

from tutorial import run_linear_tutorial


def main() -> None:
    result = run_linear_tutorial()
    output_dir = Path(__file__).with_name("outputs") / "linear_continuous_program"
    output_dir.mkdir(parents=True, exist_ok=True)
    h5ad_path = output_dir / "linear_continuous_program.h5ad"

    try:
        adata = to_anndata(result)
    except ImportError as exc:
        print(f"anndata is not installed; skipping h5ad export: {exc}")
        print(
            {
                "n_cells": int(result["layers"]["true_spliced"].shape[0]),
                "n_genes": int(result["layers"]["true_spliced"].shape[1]),
                "capture_model": result["uns"]["simulation_config"]["capture_model"],
                "alpha_source_mode": result["uns"]["simulation_config"]["alpha_source_mode"],
            }
        )
        return

    adata.write_h5ad(h5ad_path)
    print(f"saved {h5ad_path}")


if __name__ == "__main__":
    main()
