"""NVSim: lightweight GRN-aware RNA velocity simulation utilities."""

from .config import GRNConfig

# GRN schema and regulation.
from .grn import (
    GRN,
    build_graph_levels,
    calibrate_grn_half_response,
    calibrate_half_response,
    estimate_state_mean_expression,
    identify_master_regulators,
    validate_grn,
)
from .regulation import compute_alpha, hill_activation, hill_repression

# Master-regulator forcing definitions.
from .production import (
    AlphaProgram,
    StateProductionProfile,
    constant,
    linear_decrease,
    linear_increase,
    sigmoid_decrease,
    sigmoid_increase,
    transition_weight,
)

# Simulation and observed-layer generation.
from .simulate import (
    create_kinetic_vectors,
    initialize_state,
    simulate,
    simulate_bifurcation,
    simulate_linear,
    validate_positive_vector,
)
from .noise import generate_observed_counts
from .output import make_result_dict, to_anndata

# SERGIO-compatible inputs and plotting.
from .sergio_io import SergioInputs, load_sergio_targets_regs
from .plotting import (
    embed,
    plot_gene_dynamics,
    plot_gene_dynamics_over_pseudotime,
    plot_phase_portrait,
    plot_phase_gallery,
    plot_phase_portrait_gallery,
    plot_showcase,
    plot_velocity_showcase,
    prepare_adata,
    prepare_velocity_adata,
    select_genes,
    select_velocity_showcase_genes,
    run_scanpy_embedding,
    run_scvelo_velocity_stream,
    velocity_stream,
)

__all__ = [
    "GRN",
    "GRNConfig",
    "identify_master_regulators",
    "build_graph_levels",
    "estimate_state_mean_expression",
    "calibrate_grn_half_response",
    "calibrate_half_response",
    "validate_grn",
    "hill_activation",
    "hill_repression",
    "compute_alpha",
    "create_kinetic_vectors",
    "initialize_state",
    "validate_positive_vector",
    "simulate",
    "AlphaProgram",
    "StateProductionProfile",
    "constant",
    "linear_increase",
    "linear_decrease",
    "sigmoid_increase",
    "sigmoid_decrease",
    "transition_weight",
    "simulate",
    "simulate_linear",
    "simulate_bifurcation",
    "generate_observed_counts",
    "make_result_dict",
    "to_anndata",
    "SergioInputs",
    "load_sergio_targets_regs",
    "prepare_adata",
    "embed",
    "velocity_stream",
    "select_genes",
    "plot_showcase",
    "plot_phase_portrait",
    "plot_phase_gallery",
    "plot_phase_portrait_gallery",
    "plot_gene_dynamics",
    "plot_gene_dynamics_over_pseudotime",
    "prepare_velocity_adata",
    "run_scanpy_embedding",
    "run_scvelo_velocity_stream",
    "select_velocity_showcase_genes",
    "plot_velocity_showcase",
]
