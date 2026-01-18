"""
Data loading utilities and dataset definitions for Raman spectra.

Supports CSV formats with columns:
- first row optional header with wavenumber labels or separate wavenumbers array
- spectra as rows or columns with option to transpose.

Includes dataset wrappers for masked spectral modeling (MSM) and
supervised fine-tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from raman_spectra.preprocess import SpectralPreprocessor


def find_spectral_columns(df: pd.DataFrame) -> tuple[list[str], list[str], np.ndarray]:
    """
    Identifies spectral data columns by checking if the column name can be converted to a float.
    This is a robust way to separate metadata from spectral data.
    """
    spectral_cols = []
    metadata_cols = []
    for col in df.columns:
        try:
            float(col)
            spectral_cols.append(col)
        except (ValueError, TypeError):
            metadata_cols.append(col)

    wavenumbers = pd.to_numeric(spectral_cols)
    return metadata_cols, spectral_cols, wavenumbers


def load_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load spectra from CSV.

    Returns (wavenumbers, spectra_matrix, metadata_df) where spectra_matrix has shape (num_samples, num_points).
    """
    df = pd.read_csv(path)
    
    # Identify spectral columns (numeric column names) vs metadata columns
    metadata_cols, spectral_cols, wavenumbers = find_spectral_columns(df)
    
    # Extract spectral data
    spectra = df[spectral_cols].to_numpy(dtype=float)
    wn = wavenumbers.astype(float)
    metadata_df = df[metadata_cols]    
    return wn.astype(float), spectra.astype(float), metadata_df


def load_numpy(
    spectra_path: str | Path,
    wavenumbers_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load spectra from NumPy files.
    
    Parameters
    ----------
    spectra_path : str or Path
        Path to .npy file containing spectra (shape: num_samples, num_points)
    wavenumbers_path : str or Path, optional
        Path to .npy file containing wavenumbers. If None, uses indices.
        
    Returns
    -------
    tuple of np.ndarray
        (wavenumbers, spectra) where spectra has shape (num_samples, num_points)
    """
    spectra = np.load(spectra_path)
    
    if wavenumbers_path is not None:
        wavenumbers = np.load(wavenumbers_path)
    else:
        # Use indices as wavenumbers if not provided
        wavenumbers = np.arange(spectra.shape[1])
    
    return wavenumbers.astype(float), spectra.astype(float)


def load_raman_challenge_dataset(
    data_dir: str | Path,
    instruments: list[str] | None = None,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], pd.DataFrame | None]:
    """Load the dig-4-bio-raman transfer learning challenge dataset.
    
    Args:
        data_dir: Path to directory containing the CSV files
        instruments: List of instrument names to load. If None, loads all found instruments.
        
    Returns:
        Tuple of (instrument_data_dict, target_df) where:
        - instrument_data_dict: Dict mapping instrument names to (wavenumbers, spectra) tuples
        - target_df: DataFrame with target values (glucose, sodium acetate, magnesium sulfate) if available
    """
    data_dir = Path(data_dir)
    
    # Find all instrument CSV files (exclude sample_submission and transfer_plate)
    csv_files = list(data_dir.glob("*.csv"))
    special_files = {"sample_submission.csv", "transfer_plate.csv", "96_samples.csv"}
    instrument_files = [f for f in csv_files if f.name not in special_files]
    
    if instruments is not None:
        instrument_files = [f for f in instrument_files if f.stem in instruments]
    
    instrument_data = {}
    
    for csv_file in instrument_files:
        instrument_name = csv_file.stem
        try:
            wn, spectra, _ = load_csv(csv_file)
            instrument_data[instrument_name] = (wn, spectra)
            print(f"Loaded {spectra.shape[0]} spectra from {instrument_name} with {len(wn)} wavenumbers")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    # Try to load target data
    target_df = None
    target_files = ["transfer_plate.csv", "96_samples.csv"]
    for target_file in target_files:
        target_path = data_dir / target_file
        if target_path.exists():
            try:
                target_df = pd.read_csv(target_path)
                print(f"Loaded target data from {target_file} with shape {target_df.shape}")
                break
            except Exception as e:
                print(f"Error loading target file {target_file}: {e}")
    
    return instrument_data, target_df


def combine_instrument_data(
    instrument_data: dict[str, tuple[np.ndarray, np.ndarray]],
    interpolate_wavenumbers: bool = True,
    target_wavenumbers: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine data from multiple instruments into a single dataset.
    
    Args:
        instrument_data: Dict mapping instrument names to (wavenumbers, spectra) tuples
        interpolate_wavenumbers: If True, interpolate all spectra to common wavenumber grid
        target_wavenumbers: Target wavenumber grid. If None, uses the grid with most points.
        
    Returns:
        Tuple of (wavenumbers, combined_spectra, instrument_labels) where:
        - wavenumbers: Common wavenumber grid
        - combined_spectra: Stacked spectra from all instruments
        - instrument_labels: Array indicating which instrument each spectrum came from
    """
    if not instrument_data:
        raise ValueError("No instrument data provided")
    
    # Determine target wavenumber grid
    if target_wavenumbers is None:
        # Use the grid with the most points
        max_points = 0
        target_wn = None
        for wn, _ in instrument_data.values():
            if len(wn) > max_points:
                max_points = len(wn)
                target_wn = wn
        target_wavenumbers = target_wn
    
    combined_spectra = []
    instrument_labels = []
    
    for instrument_name, (wn, spectra) in instrument_data.items():
        if interpolate_wavenumbers and not np.array_equal(wn, target_wavenumbers):
            # Interpolate spectra to target wavenumber grid
            interpolated_spectra = []
            for spectrum in spectra:
                interpolated = np.interp(target_wavenumbers, wn, spectrum)
                interpolated_spectra.append(interpolated)
            spectra = np.array(interpolated_spectra)
        
        combined_spectra.append(spectra)
        instrument_labels.extend([instrument_name] * spectra.shape[0])
    
    combined_spectra = np.vstack(combined_spectra)
    instrument_labels = np.array(instrument_labels)
    
    return target_wavenumbers, combined_spectra, instrument_labels


@dataclass
class SpectraDataset(Dataset):
    wavenumbers: np.ndarray
    spectra: np.ndarray
    preprocessor: SpectralPreprocessor | None = None

    def __post_init__(self):
        if self.spectra.ndim != 2:
            raise ValueError("spectra must be 2D: (num_samples, num_points)")

    def __len__(self) -> int:
        return self.spectra.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        wn = self.wavenumbers
        y = self.spectra[idx]
        if self.preprocessor is not None:
            wn, y = self.preprocessor.apply(wn, y)
        return torch.from_numpy(wn).float(), torch.from_numpy(y).float()


class MaskedSpectraDataset(SpectraDataset):
    def __init__(
        self,
        wavenumbers: np.ndarray,
        spectra: np.ndarray,
        preprocessor: SpectralPreprocessor | None = None,
        patch_size: int = 16,
        mask_ratio: float = 0.2,
        random_state: int | None = None,
    ):
        super().__init__(wavenumbers, spectra, preprocessor)
        self.patch_size = int(patch_size)
        self.mask_ratio = float(mask_ratio)
        self.rng = np.random.default_rng(random_state)

    def __getitem__(self, idx: int):
        wn, y = super().__getitem__(idx)
        num_points = y.shape[0]
        if num_points % self.patch_size != 0:
            # center-crop to multiple of patch_size
            trimmed = (num_points // self.patch_size) * self.patch_size
            start = (num_points - trimmed) // 2
            wn = wn[start : start + trimmed]
            y = y[start : start + trimmed]
            num_points = trimmed
        num_patches = num_points // self.patch_size
        # choose mask indices
        num_mask = max(1, int(self.mask_ratio * num_patches))
        mask_idx = torch.tensor(
            self.rng.choice(num_patches, size=num_mask, replace=False), dtype=torch.long
        )
        # build a binary mask over positions
        pos_mask = torch.zeros(num_patches, dtype=torch.bool)
        pos_mask[mask_idx] = True
        return wn, y, pos_mask


@dataclass
class LabeledSpectraDataset(SpectraDataset):
    labels: np.ndarray | torch.Tensor | list[int] | list[float] | None = None

    def __getitem__(self, idx: int):
        wn, y = super().__getitem__(idx)
        if self.labels is None:
            raise ValueError("labels not provided for LabeledSpectraDataset")
        label = self.labels[idx]
        label_tensor = torch.as_tensor(label)
        return wn, y, label_tensor


