"""
NP — NeuroSolve Problem Solvers 🧠

Contient tous les modules de :
- résolution NP (subset sum, vertex cover),
- guidage EEG,
- visualisation topologique,
- XAI,
- prédiction live et interprétabilité.

Auteur : Kocupyr Romain
Licence : CC BY-NC-SA 4.0
"""

# Solveurs
from .ns001_neuro_np_solver import subset_sum_solver
from .ns002_eeg_guided_solver import eeg_guided_subset_sum

# Visualisation
from .ns003_visualizer import plot_solution_2d, plot_solution_3d
from .ns004_solver_xai import run_xai_graph_coloring

# EEG loading & preprocessing
from .ns005_eeg_real_loader import load_h5_vectors
from .ns006_live_solver import live_solver_loop
from .ns007_logger import live_session_logger

# Explorateurs mathématiques
from .ns008_riemann import map_eeg_to_riemann
from .ns008_riemann_explorer import plot_combined
from .ns009_topo_wavelet import run_wavelet_topo_analysis
from .ns010_topo_mapper import run_topo_mapping

# Évaluation
from .ns011_benchmark import run_full_benchmark
from .ns012_cycle_detector import detect_cycles

# Modèle ML
from .ns013_model_fusion import run_model
from .ns014_live_predictor import live_loop
from .ns015_shap_live import shap_explain_live

__all__ = [
    "subset_sum_solver",
    "eeg_guided_subset_sum",
    "plot_solution_2d",
    "plot_solution_3d",
    "run_xai_graph_coloring",
    "load_h5_vectors",
    "live_solver_loop",
    "live_session_logger",
    "map_eeg_to_riemann",
    "plot_combined",
    "run_wavelet_topo_analysis",
    "run_topo_mapping",
    "run_full_benchmark",
    "detect_cycles",
    "run_model",
    "live_loop",
    "shap_explain_live"
]
