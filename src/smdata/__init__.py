"""Gaussian process datasets with measured complexity, for testing GPR and KRR."""

from .complexity import (
    complexity,
    effective_dof,
    effective_rank,
    matrix_rank,
)
from .dataset import Dataset, make_dataset
from .kernel import SpectralMixture
from .means import bowl, bump, constant, ramp, step

__all__ = [
    "Dataset",
    "SpectralMixture",
    "bowl",
    "bump",
    "complexity",
    "constant",
    "effective_dof",
    "effective_rank",
    "make_dataset",
    "matrix_rank",
    "ramp",
    "step",
]
