"""Generate a scanpy/scVelo RNA velocity-style showcase for NVSim bifurcation."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_mvp_bifurcation import build_bifurcation_result

from nvsim.plotting import plot_showcase


def main() -> None:
    result = build_bifurcation_result(capture_rate=1.0, dropout_rate=0.0, poisson_observed=False)
    output_dir = Path(__file__).with_name("outputs") / "velocity_showcase"
    summary = plot_showcase(
        result,
        output_dir=output_dir,
        expression_layer="true",
        n_pcs=20,
        n_neighbors=15,
        min_dist=0.3,
        random_state=123,
    )
    print(f"saved velocity showcase to {output_dir}")
    print(summary)


if __name__ == "__main__":
    main()
