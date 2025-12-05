# metrics/__init__.py

from .metrics import detect_contradiction, semantic_similarity, evaluate_cf
from .plotting import plot_size_comparison, plot_attack_comparison

__all__ = [
    "evaluate_cf",
    "semantic_similarity",
    "detect_contradiction",
    "plot_attack_comparison",
    "plot_size_comparison"
]
