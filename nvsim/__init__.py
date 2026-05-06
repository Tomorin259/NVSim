"""NVSim: lightweight GRN-aware RNA velocity simulation utilities."""

from .config import GRNConfig, SimulationConfig, TrajectoryConfig
from .grn import GRN, validate_grn
from .regulation import hill_activation, hill_repression, compute_alpha
from .trajectory import make_bifurcation_trajectory, make_linear_trajectory
from .kinetics import create_kinetic_vectors, initialize_state, validate_positive_vector
from .programs import AlphaProgram, constant, linear_decrease, linear_increase, sigmoid_decrease, sigmoid_increase
from .simulate import simulate_bifurcation, simulate_linear
from .noise import generate_observed_counts
from .output import make_result_dict, to_anndata
from .sergio_io import SergioInputs, load_sergio_targets_regs
from .plotting import (
    compute_pca_embedding,
    compute_umap_embedding,
    plot_embedding_by_pseudotime,
    plot_embedding_by_branch,
    plot_embedding_with_velocity,
    plot_phase_portrait,
    plot_gene_dynamics_over_pseudotime,
    select_representative_genes_by_dynamics,
)

__all__ = [
    "GRN",
    "GRNConfig",
    "SimulationConfig",
    "TrajectoryConfig",
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
    "plot_gene_dynamics_over_pseudotime",
    "select_representative_genes_by_dynamics",
]
