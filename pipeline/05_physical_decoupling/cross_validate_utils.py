"""
Cross-Validation Utility Functions
====================================
Shared utilities for batch_cross_validate.py and single_cross_validate.py.

Provides:
- calculate_h_map(): 3-band Lambert-Beer height inversion
- robust_normalize(): percentile-based contrast range
"""
import numpy as np


def calculate_h_map(sample_cube, ref_cube, wavelengths, band_indices, delta_n):
    """3-band Lambert-Beer optical height inversion."""
    raw = np.array(sample_cube[:, :, band_indices])
    ref = np.array(ref_cube[:, :, band_indices])
    wavs = wavelengths[band_indices]

    T = raw / (ref + 1e-9)
    T = np.clip(T, 1e-9, 1.0)

    wavs_um = wavs / 1000.0
    k_vec = (2 * np.pi / wavs_um).reshape(1, 1, -1)

    h_cube = np.sqrt(-np.log(T)) / (k_vec * delta_n)
    h_final = np.mean(h_cube, axis=2)
    return np.squeeze(h_final)


def robust_normalize(data, percentiles=(2, 98)):
    """Compute vmin/vmax from percentile range for display contrast."""
    vmin, vmax = np.percentile(data, percentiles)
    return vmin, vmax
