"""
Raman spectra analysis package using SpecBERT and RamanSPy.

This package provides:
- Configuration management via TrainConfig
- Data loading utilities for CSV, NumPy, and challenge datasets
- Preprocessing pipelines using RamanSPy
- SpecBERT transformer model for spectral analysis
- Training and evaluation functions
"""

from .config import TrainConfig
from .data import (
    LabeledSpectraDataset,
    MaskedSpectraDataset,
    SpectraDataset,
    combine_instrument_data,
    load_csv,
    load_numpy,
    load_raman_challenge_dataset,
)
from .model import SpecBERT
from .preprocess import (
    SpectralPreprocessor,
    build_pipeline_without_normalisation,
    build_standard_pipeline,
)
from .train import evaluate_supervised, finetune_supervised, main, pretrain_msm

__all__ = [
    # Configuration
    "TrainConfig",
    # Data loading
    "load_csv",
    "load_numpy",
    "load_raman_challenge_dataset",
    "combine_instrument_data",
    # Datasets
    "SpectraDataset",
    "MaskedSpectraDataset",
    "LabeledSpectraDataset",
    # Preprocessing
    "SpectralPreprocessor",
    "build_standard_pipeline",
    "build_pipeline_without_normalisation",
    # Model
    "SpecBERT",
    # Training
    "pretrain_msm",
    "finetune_supervised",
    "evaluate_supervised",
    "main",
]
