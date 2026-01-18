"""
Configuration class for training SpecBERT on Raman spectra.

Replaces command-line arguments with a Python class-based configuration system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class TrainConfig:
    """Configuration for training SpecBERT model.
    
    Contains all parameters needed for data loading, model creation, and training.
    """
    # Training parameters
    batch_size: int = 64
    lr: float = 3e-4
    epochs: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2
    
    # Model parameters
    patch_size: int = 16
    mask_ratio: float = 0.2
    depth: int = 6
    embed_dim: int = 256
    heads: int = 8
    classes: int | None = None
    
    # Data loading parameters
    mode: Literal["pretrain", "finetune"] = "pretrain"
    input_path: str = ""
    format: Literal["csv", "npy", "challenge"] = "csv"
    wavenumbers_path: str | None = None
    labels_path: str | None = None
    instruments: list[str] | None = None
    target_column: str = "glucose"
    val_split: float = 0.0
