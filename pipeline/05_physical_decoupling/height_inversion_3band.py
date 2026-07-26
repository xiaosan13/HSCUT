"""
Simplified 3-Band Height Inversion (Lambert-Beer)
==================================================
Paper reference: Section 2.5, Methods Section 3.3 (simplified version)

Computes optical height from transmittance using a simplified
Lambert-Beer inversion on 3 selected RGB bands:

  T = I_sample / I_ref
  h = sqrt(-ln(T)) / (k * Delta_n)    where k = 2*pi / lambda

Averages across 3 bands for the final height map.
Also saves raw spectral intensity for each selected band.

Note: This is a simplified precursor to the full Cauchy+LM fitting
in cauchy_lm_fitting.py. The 3-band version uses algebraic inversion
without iterative nonlinear optimization.

Physical parameters:
- n_cell = 1.37, n_medium = 1.33, Delta_n = 0.04
- Selected bands: [122, 62, 33]

Output: Raw intensity PNGs, fused height map, scale bar
"""
import os
import sys
import argparse
import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pipeline.config_loader import load_config

DISPLAY_PERCENTILE = (2, 98)


def load_and_preprocess(hdr_path, spe_path, crop_top, rotate_180):
    if not os.path.exists(hdr_path):
        print(f"Error: file not found - {hdr_path}")
        return None, None
    try:
        img_obj = envi.open(hdr_path, spe_path)
        data = img_obj.load()
        wavelengths = np.array(img_obj.bands.centers)
        if data.shape[0] > crop_top:
            data = data[crop_top:, :, :]
        if rotate_180:
            data = np.rot90(data, k=2, axes=(0, 1))
        return data, wavelengths
    except Exception as e:
        print(f"Error loading {spe_path}: {e}")
        return None, None


def calculate_bands_and_height(sample_cube, ref_cube, wavelengths, band_indices, delta_n):
    raw_intensity = np.array(sample_cube[:, :, band_indices])
    ref_intensity = np.array(ref_cube[:, :, band_indices])
    wavs = wavelengths[band_indices]

    T = raw_intensity / (ref_intensity + 1e-9)
    T = np.clip(T, 1e-9, 1.0)

    wavs_um = wavs / 1000.0
    k_vec = (2 * np.pi / wavs_um).reshape(1, 1, -1)

    h_cube = np.sqrt(-np.log(T)) / (k_vec * delta_n)
    h_final = np.mean(h_cube, axis=2)

    return raw_intensity, h_final


def robust_normalize(data, percentiles=(2, 98)):
    vmin, vmax = np.percentile(data, percentiles)
    return vmin, vmax


def main():
    cfg = load_config()
    p = cfg['physics']

    parser = argparse.ArgumentParser(
        description='3-band Lambert-Beer height inversion from HS data')
    parser.add_argument('--sample_hdr', required=True,
                        help='Path to sample .hdr file')
    parser.add_argument('--sample_spe', required=True,
                        help='Path to sample .spe file')
    parser.add_argument('--ref_hdr', required=True,
                        help='Path to reference .hdr file')
    parser.add_argument('--ref_spe', required=True,
                        help='Path to reference .spe file')
    parser.add_argument('--output_dir', default='./output/height_inversion',
                        help='Output directory')
    parser.add_argument('--n_cell', type=float, default=p['n_cell'],
                        help='Cell refractive index')
    parser.add_argument('--n_medium', type=float, default=p['n_medium'],
                        help='Medium refractive index')
    parser.add_argument('--crop_top', type=int, default=p['crop_top'],
                        help='Rows to crop from top')
    parser.add_argument('--selected_bands', nargs='+', type=int,
                        default=p['selected_bands'],
                        help='Band indices for RGB channels')
    args = parser.parse_args()

    delta_n = args.n_cell - args.n_medium

    print(">>> Loading data...")
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

    print(f">>> Data ready. Extracting bands {args.selected_bands} and computing height...")

    raw_bands, h_map = calculate_bands_and_height(
        s_cube, r_cube, s_wavs, args.selected_bands, delta_n)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n>>> Results will be saved to: {args.output_dir}")

    for i, band_idx in enumerate(args.selected_bands):
        band_img = np.squeeze(raw_bands[:, :, i])
        vmin, vmax = robust_normalize(band_img, DISPLAY_PERCENTILE)
        filename = os.path.join(args.output_dir, f'Raw_Intensity_Band_{band_idx}.png')
        plt.imsave(filename, band_img, cmap='gray', vmin=vmin, vmax=vmax)
        print(f"   [Saved] Raw Intensity (Band {band_idx}): {filename}")

    h_map = np.squeeze(h_map)
    h_vmin, h_vmax = robust_normalize(h_map, DISPLAY_PERCENTILE)
    h_filename = os.path.join(args.output_dir, 'Height_Map_Fused.png')
    plt.imsave(h_filename, h_map, cmap='gray', vmin=h_vmin, vmax=h_vmax)
    print(f"   [Saved] Fused Height Map: {h_filename}")

    # Scale bar
    h_height_px = h_map.shape[0]
    my_dpi = 100
    target_height_px = 426.6
    fig_height_in = target_height_px / my_dpi
    fig_bar = plt.figure(figsize=(2.0, fig_height_in), dpi=my_dpi)
    ax_bar = fig_bar.add_axes([0.1, 0.02, 0.15, 0.96])
    norm = plt.Normalize(vmin=h_vmin, vmax=h_vmax)
    sm = plt.cm.ScalarMappable(cmap='gray', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_bar)
    cbar.ax.tick_params(labelsize=10)
    bar_filename = os.path.join(args.output_dir, 'Height_Map_ScaleBar.png')
    plt.savefig(bar_filename, dpi=my_dpi)
    plt.close(fig_bar)
    print(f"   [Saved] Scale Bar: {bar_filename}")

    print("\n>>> Done.")


if __name__ == "__main__":
    main()
