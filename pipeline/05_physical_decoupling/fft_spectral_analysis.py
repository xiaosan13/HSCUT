"""
FFT Spectral Analysis: SNR and Energy Distribution
====================================================
Paper reference: Section 2.5 (Figure 5A)

Analyzes the FFT of transmittance spectra along the wavelength axis:
- Extracts DC (component 0), low-frequency (component 1), and noise
  (component 5) images from FFT magnitude
- Computes energy distribution across FFT components
- Estimates SNR improvement when retaining only top K components
  (low-frequency concentration validates the FFT-based feature
   extraction used in the network input)

Input:  HS data (sample + reference)
Output: FFT component images, energy spectrum plot, SNR analysis plot
"""
import os
import sys
import argparse
import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from scipy.fft import fft

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pipeline.config_loader import load_config


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
        return data, wavelengths
    except Exception as e:
        print(f"Error: {e}")
        return None, None


def save_as_grayscale_no_contrast(data, filename):
    vmin, vmax = np.min(data), np.max(data)
    if vmax > vmin:
        norm = (data - vmin) / (vmax - vmin)
    else:
        norm = np.zeros_like(data)
    img_uint8 = (norm * 255).astype(np.uint8)
    plt.imsave(filename, img_uint8, cmap='gray')
    print(f"   [Saved] {filename}")


def estimate_snr_improvement(fft_amp):
    mean_amp_spectrum = np.mean(fft_amp, axis=(0, 1))
    energy_spectrum = mean_amp_spectrum ** 2

    n_bands = len(energy_spectrum)
    half_n = n_bands // 2

    noise_start_idx = 10
    if half_n > noise_start_idx:
        noise_floor_energy = np.mean(energy_spectrum[noise_start_idx:half_n])
    else:
        noise_floor_energy = np.min(energy_spectrum)

    total_energy = np.sum(energy_spectrum)
    total_noise_energy = n_bands * noise_floor_energy
    signal_energy = max(total_energy - total_noise_energy, 1e-10)
    raw_snr_db = 10 * np.log10(signal_energy / total_noise_energy)

    retained_energy = np.sum(energy_spectrum[:2])
    retained_signal = max(retained_energy - 2 * noise_floor_energy, 1e-10)
    filtered_noise_energy = 2 * noise_floor_energy
    filtered_snr_db = 10 * np.log10(retained_signal / filtered_noise_energy)

    return raw_snr_db, filtered_snr_db, energy_spectrum, noise_floor_energy


def main():
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description='FFT spectral analysis of HS transmittance data')
    parser.add_argument('--sample_hdr', required=True,
                        help='Path to sample .hdr file')
    parser.add_argument('--sample_spe', required=True,
                        help='Path to sample .spe file')
    parser.add_argument('--ref_hdr', required=True,
                        help='Path to reference .hdr file')
    parser.add_argument('--ref_spe', required=True,
                        help='Path to reference .spe file')
    parser.add_argument('--output_dir', default='./output/fft_analysis',
                        help='Output directory')
    parser.add_argument('--crop_top', type=int, default=cfg['physics']['crop_top'],
                        help='Rows to crop from top')
    parser.add_argument('--high_freq_index', type=int, default=5,
                        help='FFT component index to use as noise representative')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(">>> 1. Loading Data...")
    s_data, s_wavs = load_and_preprocess(args.sample_hdr, args.sample_spe,
                                         args.crop_top, True)
    r_data, r_wavs = load_and_preprocess(args.ref_hdr, args.ref_spe,
                                         args.crop_top, True)

    if s_data is None:
        return

    min_r = min(s_data.shape[0], r_data.shape[0])
    min_c = min(s_data.shape[1], r_data.shape[1])
    s_data = s_data[:min_r, :min_c, :]
    r_data = r_data[:min_r, :min_c, :]

    print(">>> 2. Calculating Transmittance (T)...")
    T_cube = s_data / (r_data + 1.0)
    T_cube = np.clip(T_cube, 0, 1.0)

    print(">>> 3. Performing FFT (Spectral Dimension)...")
    fft_cube = fft(T_cube, axis=2)
    fft_amp = np.abs(fft_cube)

    print(">>> 4. Saving Raw Component Images...")
    comp0 = fft_amp[:, :, 0]
    comp1 = fft_amp[:, :, 1]

    max_idx = fft_amp.shape[2] // 2
    noise_idx = min(args.high_freq_index, max_idx)
    comp_noise = fft_amp[:, :, noise_idx]

    save_as_grayscale_no_contrast(comp0, os.path.join(args.output_dir, 'FFT_Comp0_DC.png'))
    save_as_grayscale_no_contrast(comp1, os.path.join(args.output_dir, 'FFT_Comp1_LowFreq.png'))
    save_as_grayscale_no_contrast(comp_noise, os.path.join(args.output_dir, f'FFT_Comp{noise_idx}_Noise.png'))

    print(">>> 5. Analyzing SNR Improvement...")
    raw_snr, filtered_snr, energy_spectrum, noise_floor = estimate_snr_improvement(fft_amp)
    n_bands = len(energy_spectrum)
    half_n = n_bands // 2

    print(f"   [Result] Raw SNR: {raw_snr:.2f} dB")
    print(f"   [Result] Filtered SNR (Top 2): {filtered_snr:.2f} dB")

    indices = np.arange(1, 11)
    snr_curve = []
    for k in indices:
        curr_energy = np.sum(energy_spectrum[:k])
        curr_signal = max(curr_energy - k * noise_floor, 1e-10)
        curr_noise = k * noise_floor
        snr_val = 10 * np.log10(curr_signal / curr_noise)
        snr_curve.append(snr_val)

    # Energy spectrum plot
    fig1, ax1 = plt.subplots(figsize=(5, 4), dpi=150)
    ax1.semilogy(np.arange(half_n), energy_spectrum[:half_n],
                 color='#1f77b4', linewidth=1.5, alpha=0.9, label='Energy Spectrum')
    ax1.axhline(noise_floor, color='gray', linestyle='--', linewidth=1.5, label='Noise Floor')
    ax1.set_xlim(-0.5, 9.5)
    ax1.set_xlabel('Frequency Index (Spectral Component)', fontsize=10)
    ax1.set_ylabel('Log Energy (Amplitude$^2$)', fontsize=10)
    ax1.set_title('Spectral Energy Distribution', fontsize=12, fontweight='bold')
    ax1.axvspan(-0.5, 1.5, color='green', alpha=0.1, label='Selected Components (0-1)')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'FFT_Energy_Analysis_Plot.png'))
    plt.close(fig1)
    print("   [Saved] FFT_Energy_Analysis_Plot.png")

    # SNR improvement plot
    fig2, ax2 = plt.subplots(figsize=(5, 4), dpi=150)
    ax2.plot(indices, snr_curve, 'o-', color='#2ca02c', linewidth=2, label='Estimated SNR')
    snr_k2 = snr_curve[1]
    ax2.plot(2, snr_k2, 'r*', markersize=15, zorder=10, label='Selected (k=2)')
    ax2.axhline(raw_snr, color='gray', linestyle='--', linewidth=1.5,
                label=f'Raw SNR ({raw_snr:.1f} dB)')
    ax2.annotate(f'SNR: {snr_k2:.1f} dB\n(+{snr_k2 - raw_snr:.1f} dB)',
                 xy=(2, snr_k2), xytext=(4, snr_k2 + 1),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=10, fontweight='bold', color='#D43F3A')
    ax2.set_xlim(0, 11)
    ax2.set_xticks(np.arange(0, 12, 2))
    ax2.set_xlabel('Number of Retained Components (k)', fontsize=10)
    ax2.set_ylabel('Estimated SNR (dB)', fontsize=10)
    ax2.set_title('SNR Improvement', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'SNR_Analysis_Plot.png'))
    plt.close(fig2)
    print("   [Saved] SNR_Analysis_Plot.png")

    print(f"\n>>> Done. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
