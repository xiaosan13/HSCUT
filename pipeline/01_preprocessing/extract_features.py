"""
Feature Extraction: TrueColor Synthesis + FFT/MNF Hybrid Images
================================================================
Paper reference: Section 2.2 Lower Branch, Section 2.5 (Figure 5A)

Generates two types of features from raw hyperspectral (.spe/.hdr) data:
1. TrueColor images: Synthesized from specific spectral bands (R=122, G=62, B=33)
   with per-band 2-98% percentile contrast stretching.
2. Hybrid images: 3-channel fusion of FFT magnitude components and MNF.
   - R channel: FFT component 0 (DC / low-frequency magnitude)
   - G channel: FFT component 1 (low-frequency magnitude)
   - B channel: MNF (Minimum Noise Fraction) first component

Processing steps:
- Crop top 41 rows, rotate 180 degrees
- 3D background correction: T = clip(sample / reference, 0, 1)
- Savitzky-Golay filtering (window=15, order=3) along spectral axis
- Global MNF model trained on 10% subsampled pixels for statistical consistency
- FFT along spectral axis for frequency-domain features

Input:  Raw .spe/.hdr files in <input_dir>/<sample>/HS-DATA/
Output: TrueColor PNGs and Hybrid PNGs in <output_dir>/<sample>/
"""
import os
import gc
import glob
import traceback
import argparse
import numpy as np
import spectral.io.envi as envi
import spectral as spy
from scipy.fft import fft
from scipy.signal import savgol_filter
from PIL import Image
from tqdm import tqdm

CROP_TOP = 41
BANDS_TO_KEEP = 300
SAMPLES = ["2596995", "2596998", "2596999", "2597008"]


def get_all_tasks(input_dir):
    tasks = []
    for sample in SAMPLES:
        hs_dir = os.path.join(input_dir, sample, "HS-DATA")
        hdrs = glob.glob(os.path.join(hs_dir, "*.hdr"))
        for hdr in hdrs:
            base = os.path.splitext(os.path.basename(hdr))[0]
            spe = os.path.join(hs_dir, f"{base}.spe")
            if os.path.exists(spe):
                tasks.append((sample, base, hdr, spe))
    return tasks


def preprocess_image(memmap_data):
    data = memmap_data[CROP_TOP:, :, :BANDS_TO_KEEP].astype(np.float32)
    data = np.rot90(data, k=2, axes=(0, 1))
    return data


def main():
    parser = argparse.ArgumentParser(
        description='Extract TrueColor + MNF Hybrid features from raw HS data')
    parser.add_argument('--base_hdr', required=True,
                        help='Path to reference 0.hdr file')
    parser.add_argument('--base_spe', required=True,
                        help='Path to reference 0.spe file')
    parser.add_argument('--input_dir', required=True,
                        help='Root directory containing sample subdirectories with HS-DATA')
    parser.add_argument('--output_dir', required=True,
                        help='Root directory for output (TrueColor/Hybrid per sample)')
    args = parser.parse_args()

    all_tasks = get_all_tasks(args.input_dir)
    if not all_tasks:
        print("No files found. Check --input_dir path.")
        return

    # 1. Load and preprocess reference background data
    print(">>> Preprocessing reference background (0.spe)...")
    ref_obj = envi.open(args.base_hdr, args.base_spe)
    ref_3d = preprocess_image(ref_obj.open_memmap())
    ref_3d[ref_3d <= 0] = 1

    # Step 1: Global statistics (sample for MNF basis)
    print(f"\n>>> Step 1: Collecting global statistics ({len(all_tasks)} files total)...")
    collected_pixels = []
    sample_interval = 10

    for sample, base, hdr, spe in tqdm(all_tasks[::sample_interval], desc="Global stats", unit="file"):
        try:
            img_obj = envi.open(hdr, spe)
            raw_3d = preprocess_image(img_obj.open_memmap())
            T = np.nan_to_num(np.clip(raw_3d / ref_3d, 0, 1))
            T_smooth = savgol_filter(T, 15, 3, axis=2)
            collected_pixels.append(T_smooth[::10, ::10, :].reshape(-1, BANDS_TO_KEEP))
        except Exception:
            continue

    if not collected_pixels:
        print("Error: Statistics pool is empty.")
        return

    global_pool = np.vstack(collected_pixels)
    fake_img = global_pool.reshape(-1, 1, BANDS_TO_KEEP)
    sig_stats = spy.calc_stats(fake_img)
    noise_stats = spy.noise_from_diffs(fake_img)
    global_mnf = spy.mnf(sig_stats, noise_stats)

    mnf_sample = np.squeeze(global_mnf.reduce(fake_img, num=1))
    m_min, m_max = np.percentile(mnf_sample, (1, 99))

    del global_pool, fake_img, mnf_sample, collected_pixels
    gc.collect()

    # Step 2: Batch generation
    print("\n>>> Step 2: Applying global MNF to generate features...")

    for sample, base, hdr, spe in tqdm(all_tasks, desc="Batch processing", unit="file"):
        try:
            img_obj = envi.open(hdr, spe)
            raw_3d = preprocess_image(img_obj.open_memmap())
            T = np.nan_to_num(np.clip(raw_3d / ref_3d, 0, 1))

            def norm_8bit(b):
                p2, p98 = np.percentile(b, (2, 98))
                return np.clip((b - p2) / (p98 - p2 + 1e-6) * 255, 0, 255).astype(np.uint8)

            tc_rgb = np.dstack([norm_8bit(T[..., 122]), norm_8bit(T[..., 62]), norm_8bit(T[..., 33])])
            tc_path = os.path.join(args.output_dir, sample, "TrueColor")
            os.makedirs(tc_path, exist_ok=True)
            Image.fromarray(tc_rgb).save(os.path.join(tc_path, f"{base}_tc.png"))

            T_smooth = savgol_filter(T, 15, 3, axis=2)

            ch_b = np.squeeze(global_mnf.reduce(T_smooth, num=1))
            ch_b_norm = np.clip((ch_b - m_min) / (m_max - m_min) * 255, 0, 255).astype(np.uint8)

            f_data = fft(T_smooth, axis=2)
            ch_r = np.clip(np.abs(f_data[..., 0]) * 255, 0, 255).astype(np.uint8)
            ch_g = np.clip(np.abs(f_data[..., 1]) * 15, 0, 255).astype(np.uint8)

            hybrid_rgb = np.dstack([ch_r, ch_g, ch_b_norm])
            hy_path = os.path.join(args.output_dir, sample, "Hybrid")
            os.makedirs(hy_path, exist_ok=True)
            Image.fromarray(hybrid_rgb).save(os.path.join(hy_path, f"{base}_hybrid.png"))

        except Exception:
            continue
        finally:
            gc.collect()

    print("\n[Done] All data processed with 3D calibration + global MNF.")


if __name__ == "__main__":
    main()
