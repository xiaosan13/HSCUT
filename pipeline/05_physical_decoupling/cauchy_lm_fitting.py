"""
Physical Decoupling: Cauchy Dispersion + Levenberg-Marquardt Fitting
=====================================================================
Paper reference: Section 2.5 (Figure 5B), Methods Section 3.3
Core physical model for HSCUT's Calculation-Map.

Fits a physics-based spectral model to each pixel's transmittance
spectrum using nonlinear least squares (Levenberg-Marquardt).

Physical Model:
  I_det(lambda) = envelope * interference + diffuse

  where:
    envelope      = exp(-(k * sigma_h)^2)          # scattering attenuation
    interference  = 1 + m * cos(k * h_avg + phi_0) # coherent interference
    diffuse       = eta * (1 - envelope)           # diffuse scattering
    k             = 2*pi * Delta_n(lambda) / lambda
    Delta_n(lambda) = A + B/lambda^2               # Cauchy dispersion

  A = n_cell - n_medium = 0.04, B = 3000 (dispersion coefficient)

Options:
  --pca : Apply PCA spectral denoising before fitting (reduces noise,
          improves fitting robustness by retaining 95% variance)

Fitting Procedure (per pixel):
1. Normalize sample spectrum by reference: T = I_sample / I_ref
2. [Optional] PCA denoising on transmittance cube
3. FFT on interpolated spectrum -> peak position -> OPD -> initial h_avg
4. Parameter vector: theta = [h_avg, sigma_h, m, phi_0]
5. Levenberg-Marquardt via scipy.optimize.curve_fit with bounds
6. Parallel execution via ProcessPoolExecutor for speed

Outputs:
- h_avg map (H_avg_Macro_Map.png): Macroscopic average optical height
- sigma_h map (Sigma_h_Nano_Map.png): Nanoscale micro-roughness
- Raw band intensity images
- Scale bars

Key: h_avg and sigma_h are strictly orthogonal in the model,
enabling independent extraction of macroscopic structure and
nanoscale texture.
"""
import os
import sys
import argparse
import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fft import rfft, rfftfreq
import concurrent.futures
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pipeline.config_loader import load_config

DISPLAY_PERCENTILE = (2, 98)

# Global physics constants set after arg parsing (needed by module-level functions)
_CAUCHY_A = None
_CAUCHY_B = None


def cauchy_dispersion(lambda_nm):
    return _CAUCHY_A + _CAUCHY_B / (lambda_nm ** 2 + 1e-9)


def theoretical_spectrum(lambda_nm, h_avg, sigma_h, m, phi_0, eta=0.5):
    delta_n = cauchy_dispersion(lambda_nm)
    k_term = (2 * np.pi * delta_n) / lambda_nm
    envelope = np.exp(-(k_term * sigma_h) ** 2)
    interference = 1 + m * np.cos(k_term * h_avg + phi_0)
    diffuse = eta * (1 - envelope)
    return envelope * interference + diffuse


def estimate_initial_params(lambda_nm, spectrum):
    wavenumbers = 1.0 / lambda_nm
    k_uniform = np.linspace(wavenumbers.min(), wavenumbers.max(), len(lambda_nm))
    spectrum_uniform = np.interp(k_uniform, wavenumbers, spectrum)

    spectrum_centered = spectrum_uniform - np.mean(spectrum_uniform)
    yf = np.abs(rfft(spectrum_centered))
    xf = rfftfreq(len(k_uniform), d=(k_uniform[1] - k_uniform[0]))

    peak_idx = np.argmax(yf) if len(yf) > 0 else 0
    opd_guess = xf[peak_idx]

    mean_dn = np.mean(cauchy_dispersion(lambda_nm))
    h_avg_guess = opd_guess / mean_dn if mean_dn != 0 else 500.0

    return [abs(h_avg_guess), 10.0, 0.5, 0.0]


def fit_single_pixel(args):
    r, c, lambda_nm, R_lambda = args

    if np.mean(R_lambda) < 0.05:
        return r, c, 0.0, 0.0

    initial_guess = estimate_initial_params(lambda_nm, R_lambda)

    lower_bounds = [0, 0, 0, -np.pi]
    upper_bounds = [5000, 300, 1.0, np.pi]

    epsilon = 1e-5
    clipped_guess = np.clip(
        initial_guess,
        np.array(lower_bounds) + epsilon,
        np.array(upper_bounds) - epsilon
    )

    try:
        popt, _ = curve_fit(
            theoretical_spectrum, lambda_nm, R_lambda,
            p0=clipped_guess, bounds=(lower_bounds, upper_bounds), maxfev=1500
        )
        return r, c, popt[0], popt[1]
    except RuntimeError:
        return r, c, 0.0, 0.0


def load_and_preprocess(hdr_path, spe_path, crop_top, rotate_180):
    if not os.path.exists(hdr_path):
        return None, None
    try:
        img_obj = envi.open(hdr_path, spe_path)
        data = img_obj.load()
        wavelengths = np.array(img_obj.bands.centers)
        if data.shape[0] > crop_top:
            data = data[crop_top:, :, :]
        if rotate_180:
            data = np.rot90(data, k=2, axes=(0, 1))
        return np.array(data), wavelengths
    except Exception as e:
        print(f"Error loading {spe_path}: {e}")
        return None, None


def robust_normalize(data, percentiles=(2, 98)):
    vmin, vmax = np.percentile(data[data > 0], percentiles) if np.any(data > 0) else (0, 1)
    return vmin, vmax


def save_scale_bar(vmin, vmax, output_dir, filename, cmap='gray', my_dpi=100, target_height_px=426.6):
    fig_height_in = target_height_px / my_dpi
    fig_bar = plt.figure(figsize=(2.0, fig_height_in), dpi=my_dpi)
    ax_bar = fig_bar.add_axes([0.1, 0.02, 0.15, 0.96])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_bar)
    cbar.ax.tick_params(labelsize=10)
    bar_path = os.path.join(output_dir, filename)
    plt.savefig(bar_path, dpi=my_dpi)
    plt.close(fig_bar)
    print(f"   [Saved] Scale Bar: {bar_path}")


def apply_pca_denoising(T_cube, variance_ratio):
    """PCA spectral denoising on transmittance cube."""
    from sklearn.decomposition import PCA
    H, W, Bands = T_cube.shape
    print(f"    -> Reshaping to (pixels={H * W}, bands={Bands})...")

    X = T_cube.reshape(H * W, Bands)

    print(f"    -> Running PCA (target variance: {variance_ratio * 100:.0f}%)...")
    pca = PCA(n_components=variance_ratio)
    X_pca = pca.fit_transform(X)

    n_selected = pca.n_components_
    explained = np.sum(pca.explained_variance_ratio_) * 100
    print(f"    -> PCA done. Selected {n_selected} components, retaining {explained:.2f}% variance.")

    print("    -> Inverse transform to reconstruct denoised spectra...")
    X_reconstructed = pca.inverse_transform(X_pca)

    T_cube_denoised = X_reconstructed.reshape(H, W, Bands)
    T_cube_denoised = np.clip(T_cube_denoised, 1e-9, 1.2)
    return T_cube_denoised


def main():
    global _CAUCHY_A, _CAUCHY_B

    cfg = load_config()
    p = cfg['physics']

    parser = argparse.ArgumentParser(
        description='Cauchy+LM physical decoupling: extract h_avg and sigma_h from HS data')
    parser.add_argument('--sample_hdr', required=True,
                        help='Path to sample .hdr file')
    parser.add_argument('--sample_spe', required=True,
                        help='Path to sample .spe file')
    parser.add_argument('--ref_hdr', required=True,
                        help='Path to reference .hdr file')
    parser.add_argument('--ref_spe', required=True,
                        help='Path to reference .spe file')
    parser.add_argument('--output_dir', default='./output/cauchy_lm_fitting',
                        help='Output directory')
    parser.add_argument('--n_cell', type=float, default=p['n_cell'],
                        help='Cell refractive index')
    parser.add_argument('--n_medium', type=float, default=p['n_medium'],
                        help='Medium refractive index')
    parser.add_argument('--cauchy_a', type=float, default=p['cauchy_a'],
                        help='Cauchy dispersion A coefficient')
    parser.add_argument('--cauchy_b', type=float, default=p['cauchy_b'],
                        help='Cauchy dispersion B coefficient')
    parser.add_argument('--crop_top', type=int, default=p['crop_top'],
                        help='Rows to crop from top')
    parser.add_argument('--selected_bands', nargs='+', type=int,
                        default=p['selected_bands'],
                        help='Band indices for raw intensity output')
    parser.add_argument('--max_workers', type=int, default=max(1, os.cpu_count() - 4),
                        help='Number of parallel workers')
    parser.add_argument('--pca', action='store_true',
                        help='Apply PCA spectral denoising before fitting')
    parser.add_argument('--pca_variance', type=float, default=0.95,
                        help='Variance ratio to retain in PCA denoising (default: 0.95)')
    args = parser.parse_args()

    _CAUCHY_A = args.cauchy_a
    _CAUCHY_B = args.cauchy_b

    print(">>> Loading full-band hyperspectral data...")
    s_cube, s_wavs = load_and_preprocess(args.sample_hdr, args.sample_spe,
                                         args.crop_top, True)
    r_cube, r_wavs = load_and_preprocess(args.ref_hdr, args.ref_spe,
                                         args.crop_top, True)

    if s_cube is None or r_cube is None:
        return

    min_r = min(s_cube.shape[0], r_cube.shape[0])
    min_c = min(s_cube.shape[1], r_cube.shape[1])
    s_cube = s_cube[:min_r, :min_c, :]
    r_cube = r_cube[:min_r, :min_c, :]

    os.makedirs(args.output_dir, exist_ok=True)

    # Save raw intensity bands
    print("\n>>> Extracting and saving raw intensity bands...")
    raw_intensity_bands = s_cube[:, :, args.selected_bands]
    for i, band_idx in enumerate(args.selected_bands):
        band_img = np.squeeze(raw_intensity_bands[:, :, i])
        vmin, vmax = robust_normalize(band_img, DISPLAY_PERCENTILE)
        filename = os.path.join(args.output_dir, f'Raw_Intensity_Band_{band_idx}.png')
        plt.imsave(filename, band_img, cmap='gray', vmin=vmin, vmax=vmax)
        print(f"   [Saved] {filename}")

    # Compute transmittance
    print("\n>>> Computing transmittance / relative reflectance...")
    T_cube = s_cube / (r_cube + 1e-9)
    T_cube = np.clip(T_cube, 1e-9, 1.2)

    # Optional PCA denoising
    if args.pca:
        print("\n>>> [PCA] Applying spectral denoising...")
        T_cube = apply_pca_denoising(T_cube, variance_ratio=args.pca_variance)
        print("    -> PCA denoising complete.")

    h_avg_map = np.zeros((min_r, min_c))
    sigma_h_map = np.zeros((min_r, min_c))

    print(f"\n>>> Starting full-band physical model decoupling (h_avg + sigma_h)...")
    print(f"    Image size: {min_r} x {min_c}, workers: {args.max_workers}")
    start_time = time.time()

    tasks = []
    for r in range(min_r):
        for c in range(min_c):
            tasks.append((r, c, s_wavs, T_cube[r, c, :]))

    completed = 0
    total_tasks = len(tasks)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        for result in executor.map(fit_single_pixel, tasks, chunksize=min_c):
            r, c, h_avg, sigma_h = result
            h_avg_map[r, c] = h_avg
            sigma_h_map[r, c] = sigma_h

            completed += 1
            if completed % (total_tasks // 10 + 1) == 0:
                print(f"    Progress: {completed}/{total_tasks} ({100 * completed / total_tasks:.1f}%)")

    print(f"    Decoupling complete! Elapsed: {time.time() - start_time:.2f}s")

    # Save h_avg map
    print("\n>>> Saving decoupled feature maps...")
    h_avg_vmin, h_avg_vmax = robust_normalize(h_avg_map, DISPLAY_PERCENTILE)
    h_avg_filename = os.path.join(args.output_dir, 'H_avg_Macro_Map.png')
    plt.imsave(h_avg_filename, h_avg_map, cmap='gray', vmin=h_avg_vmin, vmax=h_avg_vmax)
    print(f"   [Saved] h_avg: {h_avg_filename}")
    save_scale_bar(h_avg_vmin, h_avg_vmax, args.output_dir, 'H_avg_ScaleBar.png')

    # Save sigma_h map
    sigma_h_vmin, sigma_h_vmax = robust_normalize(sigma_h_map, DISPLAY_PERCENTILE)
    sigma_h_filename = os.path.join(args.output_dir, 'Sigma_h_Nano_Map.png')
    plt.imsave(sigma_h_filename, sigma_h_map, cmap='magma', vmin=sigma_h_vmin, vmax=sigma_h_vmax)
    print(f"   [Saved] sigma_h: {sigma_h_filename}")
    save_scale_bar(sigma_h_vmin, sigma_h_vmax, args.output_dir, 'Sigma_h_ScaleBar.png', cmap='magma')

    print("\n>>> All tasks complete.")


if __name__ == "__main__":
    main()
