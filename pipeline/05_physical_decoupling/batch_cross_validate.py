"""
Batch Cross-Validation: Height Map vs Virtual Stain (Multi-Sample)
====================================================================
Paper reference: Section 2.5

Batch processing version of single_cross_validate.py.
Iterates over ALL virtual stain PNGs in a directory, computes:

- Pearson R: pixel-wise correlation between h_avg and nuclear density
- Dice coefficient: spatial overlap between thresholded maps

Processing:
- Trims top/bottom N samples by R value (TRIM_COUNT=10)
- Selects best sample for detailed visualization
- Saves all metrics to batch_metrics.csv

Input:  Virtual stain directory + corresponding HS data
Output: batch_metrics.csv, Best_Sample_H_Map.png, scale bar
"""
import os
import sys
import glob
import argparse
import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from skimage import io, transform
from skimage.color import rgb2hed
from scipy.stats import pearsonr
import time
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline.config_loader import load_config
from cross_validate_utils import calculate_h_map, robust_normalize

TRIM_COUNT = 10
DISPLAY_PERCENTILE = (2, 98)


def parse_filename(filename):
    name_no_ext = os.path.splitext(filename)[0]
    sample_name, separator, file_id = name_no_ext.rpartition('_')
    if not separator:
        return None, None
    return sample_name, file_id


def load_hs_data(sample_name, file_id, hs_data_root, crop_top, rotate_180):
    sample_dir = os.path.join(hs_data_root, sample_name)
    s_hdr = os.path.join(sample_dir, f"{file_id}.hdr")
    s_spe = os.path.join(sample_dir, f"{file_id}.spe")
    r_hdr = os.path.join(sample_dir, "0.hdr")
    r_spe = os.path.join(sample_dir, "0.spe")

    if not (os.path.exists(s_hdr) and os.path.exists(r_hdr)):
        return None, None, None

    try:
        s_obj = envi.open(s_hdr, s_spe)
        s_data = s_obj.load()
        r_obj = envi.open(r_hdr, r_spe)
        r_data = r_obj.load()
        wavelengths = np.array(s_obj.bands.centers)

        if s_data.shape[0] > crop_top:
            s_data = s_data[crop_top:, :, :]
        if r_data.shape[0] > crop_top:
            r_data = r_data[crop_top:, :, :]

        if rotate_180:
            s_data = np.rot90(s_data, k=2, axes=(0, 1))
            r_data = np.rot90(r_data, k=2, axes=(0, 1))

        min_r = min(s_data.shape[0], r_data.shape[0])
        min_c = min(s_data.shape[1], r_data.shape[1])
        s_data = s_data[:min_r, :min_c, :]
        r_data = r_data[:min_r, :min_c, :]

        return s_data, r_data, wavelengths
    except Exception as e:
        print(f"Error loading {sample_name}/{file_id}: {e}")
        return None, None, None


def calculate_dice_coefficient(h_map, nuclei_density, percentile=80):
    thresh_h = np.percentile(h_map, percentile)
    mask_h = h_map > thresh_h

    thresh_nuc = np.percentile(nuclei_density, percentile)
    mask_nuc = nuclei_density > thresh_nuc

    intersection = np.logical_and(mask_h, mask_nuc).sum()
    size_h = mask_h.sum()
    size_nuc = mask_nuc.sum()

    if size_h + size_nuc == 0:
        return 0.0

    dice = 2.0 * intersection / (size_h + size_nuc)
    return dice


def process_single_sample(png_path, hs_data_root, crop_top, rotate_180,
                          selected_bands, delta_n):
    filename = os.path.basename(png_path)
    sample_name, file_id = parse_filename(filename)
    if not sample_name:
        return None

    try:
        gt_img = io.imread(png_path)
        if gt_img.shape[-1] == 4:
            gt_img = gt_img[:, :, :3]
        target_shape = gt_img.shape[:2]
    except Exception:
        return None

    s_data, r_data, wavs = load_hs_data(sample_name, file_id, hs_data_root,
                                        crop_top, rotate_180)
    if s_data is None:
        return None

    h_map = calculate_h_map(s_data, r_data, wavs, selected_bands, delta_n)
    h_resized = transform.resize(h_map, target_shape, order=1, mode='reflect',
                                 anti_aliasing=True, preserve_range=True)

    hed = rgb2hed(gt_img)
    nuclei_ch = hed[:, :, 0]
    nuclei_density = (nuclei_ch - nuclei_ch.min()) / (nuclei_ch.max() - nuclei_ch.min())

    r_val, _ = pearsonr(h_resized.flatten(), nuclei_density.flatten())
    dice_val = calculate_dice_coefficient(h_resized, nuclei_density, percentile=80)

    return {
        'filename': filename,
        'r_val': r_val,
        'dice_val': dice_val,
        'h_map': h_resized,
        'nuclei_density': nuclei_density,
        'target_shape': target_shape
    }


def save_scale_bar(output_dir, vmin, vmax, height_px):
    my_dpi = 100
    fig_height_in = height_px / my_dpi
    fig_bar = plt.figure(figsize=(2.0, fig_height_in), dpi=my_dpi)
    ax_bar = fig_bar.add_axes([0.1, 0.02, 0.15, 0.96])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap='gray', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_bar)
    cbar.ax.tick_params(labelsize=10)
    plt.savefig(os.path.join(output_dir, 'Best_Sample_ScaleBar.png'), dpi=my_dpi)
    plt.close(fig_bar)


def main():
    cfg = load_config()
    p = cfg['physics']

    parser = argparse.ArgumentParser(
        description='Batch cross-validation: height map vs virtual stain')
    parser.add_argument('--virtual_stain_dir', required=True,
                        help='Directory containing virtual stain PNGs (fake_B)')
    parser.add_argument('--hs_data_root', default=cfg['raw_data']['hs_root'],
                        help='Root directory for raw HS data')
    parser.add_argument('--output_dir', default='./output/batch_cross_validate',
                        help='Output directory')
    parser.add_argument('--n_cell', type=float, default=p['n_cell'],
                        help='Cell refractive index')
    parser.add_argument('--n_medium', type=float, default=p['n_medium'],
                        help='Medium refractive index')
    parser.add_argument('--crop_top', type=int, default=p['crop_top'],
                        help='Rows to crop from top')
    parser.add_argument('--selected_bands', nargs='+', type=int,
                        default=p['selected_bands'],
                        help='Band indices for height inversion')
    args = parser.parse_args()

    delta_n = args.n_cell - args.n_medium
    os.makedirs(args.output_dir, exist_ok=True)

    png_files = glob.glob(os.path.join(args.virtual_stain_dir, "*.png"))
    total_files = len(png_files)
    print(f">>> Found {total_files} samples. Starting analysis (Metrics: R & Dice)...")

    all_results = []
    start_time = time.time()

    for idx, png_path in enumerate(png_files):
        print(f"[{idx + 1}/{total_files}] Processing {os.path.basename(png_path)}...", end="\r")
        res = process_single_sample(png_path, args.hs_data_root, args.crop_top,
                                    True, args.selected_bands, delta_n)
        if res:
            all_results.append(res)

    print(f"\n>>> Processing complete! Elapsed: {time.time() - start_time:.1f}s")
    print(f"    Valid samples: {len(all_results)}")

    if len(all_results) < (TRIM_COUNT * 2 + 1):
        print(f"Error: Not enough samples for trimming")
        return

    sorted_results = sorted(all_results, key=lambda x: x['r_val'])
    filtered_results = sorted_results[TRIM_COUNT:-TRIM_COUNT]

    n_samples = len(filtered_results)
    print(f"    After trimming: {n_samples} samples")

    best_data = filtered_results[-1]
    best_r = best_data['r_val']
    print(f"    Best sample: {best_data['filename']} (r={best_r:.4f}, dice={best_data['dice_val']:.4f})")

    # Save CSV
    csv_path = os.path.join(args.output_dir, 'batch_metrics.csv')
    print(f">>> Exporting data to: {csv_path}")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename', 'Pearson_R', 'Dice_Coefficient'])
        for res in filtered_results:
            writer.writerow([res['filename'], f"{res['r_val']:.5f}", f"{res['dice_val']:.5f}"])
    print(f"   [Saved] batch_metrics.csv")

    # Save best sample H-map
    h_best = best_data['h_map']
    h_vmin, h_vmax = robust_normalize(h_best, (2, 98))

    h_map_path = os.path.join(args.output_dir, 'Best_Sample_H_Map.png')
    plt.imsave(h_map_path, h_best, cmap='gray', vmin=h_vmin, vmax=h_vmax)
    print(f"   [Saved] Best Sample H-map: {h_map_path}")

    save_scale_bar(args.output_dir, h_vmin, h_vmax, best_data['target_shape'][0])
    print(f"   [Saved] Best Sample Scale Bar")

    print(f"\n>>> Done. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
