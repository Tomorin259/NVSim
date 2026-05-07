"""NVSim: lightweight GRN-aware RNA velocity simulation utilities."""

from .config import GRNConfig, SimulationConfig, TrajectoryConfig

# GRN schema and regulation.
from .grn import (
    GRN,
    build_graph_levels,
    calibrate_grn_thresholds,
    calibrate_half_response,
    estimate_state_mean_expression,
    identify_master_regulators,
    validate_grn,
)
from .regulation import compute_alpha, hill_activation, hill_repression

# Trajectory and kinetic utilities.
from .kinetics import create_kinetic_vectors, initialize_state, validate_positive_vector
from .trajectory import make_bifurcation_trajectory, make_linear_trajectory

# Master-regulator forcing definitions.
from .production import (
    AlphaProgram,
    StateProductionProfile,
    constant,
    linear_decrease,
    linear_increase,
    sigmoid_decrease,
    sigmoid_increase,
)

# Simulation and observed-layer generation.
from .simulate import simulate_bifurcation, simulate_linear
from .noise import generate_observed_counts
from .output import make_result_dict, to_anndata

# SERGIO-compatible inputs and quick-look plotting.
from .sergio_io import SergioInputs, load_sergio_targets_regs
from .plotting import (
    compute_pca_embedding,
    compute_umap_embedding,
    plot_embedding_by_branch,
    plot_embedding_by_pseudotime,
    plot_embedding_with_velocity,
    plot_gene_dynamics_over_pseudotime,
    plot_phase_portrait,
    plot_phase_portrait_gallery,
    select_representative_genes_by_dynamics,
)

__all__ = [
    "GRN",
    "GRNConfig",
    "SimulationConfig",
    "TrajectoryConfig",
    "identify_master_regulators",
    "build_graph_levels",
    "estimate_state_mean_expression",
    "calibrate_grn_thresholds",
    "calibrate_half_response",
    "validate_grn",
    "hill_activation",
    "hill_repression",
    "compute_alpha",
    "make_linear_trajectory",
    "make_bifurcation_trajectory",
    "create_kinetic_vectors",
    "initialize_state",
    "validate_positive_vector",
    "AlphaProgram",
    "StateProductionProfile",
    "constant",
    "linear_increase",
    "linear_decrease",
    "sigmoid_increase",
    "sigmoid_decrease",
    "simulate_linear",
    "simulate_bifurcation",
    "generate_observed_counts",
    "make_result_dict",
    "to_anndata",
    "SergioInputs",
    "load_sergio_targets_regs",
    "compute_pca_embedding",
    "compute_umap_embedding",
    "plot_embedding_by_pseudotime",
    "plot_embedding_by_branch",
    "plot_embedding_with_velocity",
    "plot_phase_portrait",
    "plot_phase_portrait_gallery",
    "plot_gene_dynamics_over_pseudotime",
    "select_representative_genes_by_dynamics",
]
