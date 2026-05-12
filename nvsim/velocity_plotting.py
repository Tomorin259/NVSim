"""Compatibility shim for the former velocity plotting module.

New code should import from ``nvsim.plotting``. This module keeps older
``nvsim.velocity_plotting`` imports working while the public plotting API is
consolidated into a single module.
"""

from .plotting import (
    embed as run_scanpy_embedding,
    plot_showcase as plot_velocity_showcase,
    prepare_adata as prepare_velocity_adata,
    select_genes as select_velocity_showcase_genes,
    velocity_stream as run_scvelo_velocity_stream,
)

__all__ = [
    "prepare_velocity_adata",
    "run_scanpy_embedding",
    "run_scvelo_velocity_stream",
    "select_velocity_showcase_genes",
    "plot_velocity_showcase",
]
