"""
Preprocessing utilities for Raman spectra using RamanSPy.

Provides wrappers and helpers for building ramanspy preprocessing pipelines
and adapting them for use with the dataset classes.
"""

from __future__ import annotations


import numpy as np
import ramanspy as rp

MIN_WAVENUMBER = 300
MAX_WAVENUMBER = 1942


def build_standard_pipeline(
    crop_region: tuple[float, float] | None = (MIN_WAVENUMBER, MAX_WAVENUMBER),
    normalize: bool = False,
) -> rp.preprocessing.Pipeline:
    """Build a standard RamanSPy preprocessing pipeline.
    
    Pipeline includes:
    - Cropping to fingerprint region (optional)
    - Whitaker-Hayes cosmic ray removal
    - Savitzky-Golay smoothing (window=9, polyorder=3)
    - ASPLS baseline correction
    - MinMax normalization (optional)
    
    Parameters
    ----------
    crop_region : tuple of float, optional
        (min_wavenumber, max_wavenumber) for cropping. If None, no cropping.
    normalize : bool, default False
        Whether to include MinMax normalization in the pipeline.
        
    Returns
    -------
    rp.preprocessing.Pipeline
        Configured preprocessing pipeline
    """
    steps = []
    
    if crop_region is not None:
        steps.append(rp.preprocessing.misc.Cropper(region=crop_region))
    
    steps.extend([
        rp.preprocessing.despike.WhitakerHayes(),
        rp.preprocessing.denoise.SavGol(window_length=9, polyorder=3),
        rp.preprocessing.baseline.ASPLS(),
    ])
    
    if normalize:
        steps.append(rp.preprocessing.normalise.MinMax())
    
    return rp.preprocessing.Pipeline(steps)


def build_pipeline_without_normalisation(
    crop_region: tuple[float, float] | None = (MIN_WAVENUMBER, MAX_WAVENUMBER),
) -> rp.preprocessing.Pipeline:
    """Build a standard preprocessing pipeline without normalization.
    
    Useful for regression tasks where normalization may not be desired.
    
    Parameters
    ----------
    crop_region : tuple of float, optional
        (min_wavenumber, max_wavenumber) for cropping. If None, no cropping.
        
    Returns
    -------
    rp.preprocessing.Pipeline
        Configured preprocessing pipeline without normalization
    """
    return build_standard_pipeline(crop_region=crop_region, normalize=False)


class SpectralPreprocessor:
    """Adapter wrapper for ramanspy preprocessing pipelines.
    
    Provides a compatible interface for the existing dataset classes
    that expect a preprocessor with an `apply(wavenumbers, spectrum)` method.
    
    This class wraps a ramanspy Pipeline and adapts it to work with
    numpy arrays and the existing dataset interface.
    """
    
    def __init__(self, pipeline: rp.preprocessing.Pipeline | None = None):
        """Initialize the preprocessor.
        
        Parameters
        ----------
        pipeline : rp.preprocessing.Pipeline, optional
            RamanSPy preprocessing pipeline. If None, uses default standard pipeline.
        """
        if pipeline is None:
            pipeline = build_standard_pipeline()
        self.pipeline = pipeline
    
    def apply(self, wavenumbers: np.ndarray, spectrum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply preprocessing pipeline to a spectrum.
        
        Parameters
        ----------
        wavenumbers : np.ndarray
            Wavenumber axis (1D array)
        spectrum : np.ndarray
            Spectrum intensities (1D array)
            
        Returns
        -------
        tuple of np.ndarray
            (processed_wavenumbers, processed_spectrum)
        """
        # Create a ramanspy Spectrum object
        raman_spectrum = rp.Spectrum(spectrum, wavenumbers)
        
        # Apply preprocessing pipeline
        processed_spectrum = self.pipeline.apply(raman_spectrum)
        
        # Extract processed data
        # ramanspy Spectrum objects have spectral_axis and spectral_data attributes
        # spectral_axis is the wavenumber axis, spectral_data is the intensity array
        if hasattr(processed_spectrum, 'spectral_axis'):
            processed_wn = processed_spectrum.spectral_axis
        else:
            # Fallback: use original wavenumbers if not available
            processed_wn = wavenumbers
        
        if hasattr(processed_spectrum, 'spectral_data'):
            # spectral_data is already a numpy array
            processed_intensities = processed_spectrum.spectral_data
        else:
            # Fallback: try 'data' attribute or use original spectrum
            processed_intensities = getattr(processed_spectrum, 'data', spectrum)
        
        # Ensure we return numpy arrays
        if not isinstance(processed_wn, np.ndarray):
            processed_wn = np.array(processed_wn)
        if not isinstance(processed_intensities, np.ndarray):
            processed_intensities = np.array(processed_intensities)
        
        return processed_wn, processed_intensities