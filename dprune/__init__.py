"""
dPrune: A lightweight, extensible Python library for data pruning with Hugging Face datasets and transformers.

This library provides tools for data pruning through a modular framework of Scorers and Pruners.
"""

from .base import Scorer, Pruner
from .pipeline import PruningPipeline
from .callbacks import ForgettingCallback

# Import main scorer classes
from .scorers.supervised import CrossEntropyScorer, ForgettingScorer
from .scorers.unsupervised import KMeansCentroidDistanceScorer

# Import main pruner classes
from .pruners.selection import TopKPruner, BottomKPruner, StratifiedPruner, RandomPruner

__version__ = "0.0.1"
__all__ = [
    # Base classes
    "Scorer",
    "Pruner",
    # Core components
    "PruningPipeline",
    "ForgettingCallback",
    # Scorers
    "CrossEntropyScorer",
    "ForgettingScorer", 
    "KMeansCentroidDistanceScorer",
    # Pruners
    "TopKPruner",
    "BottomKPruner",
    "StratifiedPruner",
    "RandomPruner",
]
