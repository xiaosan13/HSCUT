"""
Single-Sample Cross-Validation: Height Map vs Virtual Stain
=============================================================
Paper reference: Section 2.5 (Figure 5B)

Validates the physical height map (h_avg) against virtual H&E staining
results for a single sample:

1. Aligns height map to virtual stain image dimensions
2. Extracts H&E hematoxylin channel (HED deconvolution) as nuclear GT
3. Computes Pearson correlation (r) between height map and nuclear density
4. Generates:
   - Hexbin correlation scatter plot
   - Violin distribution plot (background vs nuclei height values)
   - Dual versions: with/without axis ticks for publication

Input:  HS data + virtual stain image from CUT output
Output: Correlation hexbin, violin distribution, aligned height map,
        GT nuclei density map, scale bars
"""
import os
import sys
import argparse
import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from skimage import io, transform
from skimage.color import rgb2hed
from scipy.stats import pearsonr

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline.config_loader import load_config
from cross_validate_utils import calculate_h_map, robust_normalize

DISPLAY_PERCENTILE = (2, 98)


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
        print(f"Error loading HS data: {e}")
        return None, None


def save_scale_bar(output_dir, vmin, vmax, height_px):
    my_dpi = 100
    fig_height_in = height_px / my_dpi
    fig_bar = plt.figure(figsize=(2.0, fig_height_in), dpi=my_dpi)
    ax_bar = fig_bar.add_axes([0.1, 0.02, 0.15, 0.96])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap='gray', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_bar)
    cbar.ax.tick_params(labelsize=12)
    bar_filename = os.path.join(output_dir, f'ScaleBar_{height_px}px.png')
    plt.savefig(bar_filename, dpi=my_dpi)
    plt.close(fig_bar)


def plot_hexbin_correlation(flat_h, flat_nuclei, h_vmin, h_vmax, output_dir, filename, show_ticks=True):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    hb = ax.hexbin(flat_h, flat_nuclei, gridsize=50, cmap='inferno', mincnt=1, bins='log')
    cb = fig.colorbar(hb, ax=ax)

    ax.set_xlabel('', fontsize=0)
    ax.set_ylabel('', fontsize=0)
    ax.set_title('', fontsize=0)
    cb.set_label('', fontsize=0)

    ax.set_xlim(h_vmin, h_vmax)

    if not show_ticks:
        ax.tick_params(axis='both', which='both', bottom=False, left=False,
                       labelbottom=False, labelleft=False)
        cb.ax.tick_params(axis='y', which='both', length=0, labelright=False)
    else:
        ax.tick_params(axis='both', which='major', labelsize=12)
        cb.ax.tick_params(labelsize=12)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    plt.grid(True, linestyle='-', color='#eeeeee', alpha=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close(fig)


def plot_violin_distribution(h_vals_bg, h_vals_nuclei, output_dir, filename, show_ticks=True):
    fig, ax = plt.subplots(figsize=(4, 5), dpi=300)
    parts = ax.violinplot([h_vals_bg, h_vals_nuclei], showmeans=True, showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['', ''])
    ax.set_ylabel('', fontsize=0)
    ax.set_title('', fontsize=0)

    if not show_ticks:
        ax.tick_params(axis='both', which='both', bottom=False, left=False,
                       labelbottom=False, labelleft=False)
    else:
        ax.tick_params(axis='y', which='major', labelsize=12)
        ax.tick_params(axis='x', which='both', bottom=True)

    mu1, mu2 = np.mean(h_vals_bg), np.mean(h_vals_nuclei)
    font_prop = {'family': 'Microsoft YaHei', 'weight': 'bold', 'size': 18}
    ax.text(1, mu1, f'{mu1:.2f}', ha='center', va='bottom', fontdict=font_prop)
    ax.text(2, mu2, f'{mu2:.2f}', ha='center', va='bottom', fontdict=font_prop)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    plt.grid(True, linestyle='-', color='#eeeeee', alpha=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close(fig)


def validate_and_export(h_map, virtual_path, output_dir):
    print("\n>>> Starting alignment and validation analysis...")

    if not os.path.exists(virtual_path):
        print(f"   [Error] Virtual stain image not found: {virtual_path}")
        return

    try:
        virtual_rgb = io.imread(virtual_path)
        if virtual_rgb.shape[-1] == 4:
            virtual_rgb = virtual_rgb[:, :, :3]
        target_shape = virtual_rgb.shape[:2]
    except Exception as e:
        print(f"   [Error] Failed to read virtual stain image: {e}")
        return

    h_map_resized = transform.resize(h_map, target_shape, order=1, mode='reflect',
                                     anti_aliasing=True, preserve_range=True)
    h_vmin, h_vmax = robust_normalize(h_map_resized, DISPLAY_PERCENTILE)

    plt.imsave(os.path.join(output_dir, 'H_Map_Resized_Aligned.png'),
               h_map_resized, cmap='gray', vmin=h_vmin, vmax=h_vmax)
    save_scale_bar(output_dir, h_vmin, h_vmax, height_px=target_shape[0])

    hed = rgb2hed(virtual_rgb)
    h_channel = hed[:, :, 0]
    h_nuclei_density = (h_channel - h_channel.min()) / (h_channel.max() - h_channel.min())
    io.imsave(os.path.join(output_dir, 'Validation_GT_Nuclei.png'),
              (h_nuclei_density * 255).astype(np.uint8))

    flat_h = h_map_resized.flatten()
    flat_nuclei = h_nuclei_density.flatten()
    corr, _ = pearsonr(flat_h, flat_nuclei)
    print(f"   [Result] Pearson Correlation (r): {corr:.4f}")

    threshold_nuclei = np.percentile(h_nuclei_density, 80)
    mask_nuclei = h_nuclei_density > threshold_nuclei
    mask_bg = ~mask_nuclei

    h_vals_nuclei = h_map_resized[mask_nuclei]
    h_vals_bg = h_map_resized[mask_bg]

    sample_size = 5000
    if len(h_vals_nuclei) > sample_size:
        h_vals_nuclei = np.random.choice(h_vals_nuclei, sample_size, replace=False)
    if len(h_vals_bg) > sample_size:
        h_vals_bg = np.random.choice(h_vals_bg, sample_size, replace=False)

    print("   Generating figures (dual versions)...")
    plot_hexbin_correlation(flat_h, flat_nuclei, h_vmin, h_vmax, output_dir,
                            'Correlation_Plot_with_ticks.png', show_ticks=True)
    plot_violin_distribution(h_vals_bg, h_vals_nuclei, output_dir,
                             'Distribution_Plot_with_ticks.png', show_ticks=True)
    plot_hexbin_correlation(flat_h, flat_nuclei, h_vmin, h_vmax, output_dir,
                            'Correlation_Plot_no_ticks.png', show_ticks=False)
    plot_violin_distribution(h_vals_bg, h_vals_nuclei, output_dir,
                             'Distribution_Plot_no_ticks.png', show_ticks=False)

    print(f"   [Saved] All figures to: {output_dir}")


def main():
    cfg = load_config()
    p = cfg['physics']

    parser = argparse.ArgumentParser(
        description='Single-sample cross-validation: height map vs virtual stain')
    parser.add_argument('--sample_hdr', required=True,
                        help='Path to sample .hdr file')
    parser.add_argument('--sample_spe', required=True,
                        help='Path to sample .spe file')
    parser.add_argument('--ref_hdr', required=True,
                        help='Path to reference .hdr file')
    parser.add_argument('--ref_spe', required=True,
                        help='Path to reference .spe file')
    parser.add_argument('--virtual_stain', required=True,
                        help='Path to virtual stain PNG image')
    parser.add_argument('--output_dir', default='./output/single_cross_validate',
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

    print(">>> Loading HS data...")
    s_cube, s_wavs = load_and_preprocess(args.sample_hdr, args.sample_spe,
                                         args.crop_top, True)
    r_cube, r_wavs = load_and_preprocess(args.ref_hdr, args.ref_spe,
                                         args.crop_top, True)
    if s_cube is None:
        return

    min_r = min(s_cube.shape[0], r_cube.shape[0])
    min_c = min(s_cube.shape[1], r_cube.shape[1])
    s_cube = s_cube[:min_r, :min_c, :]
    r_cube = r_cube[:min_r, :min_c, :]

    print(f">>> Computing height map (original size)...")
    h_map = calculate_h_map(s_cube, r_cube, s_wavs, args.selected_bands, delta_n)

    os.makedirs(args.output_dir, exist_ok=True)
    validate_and_export(h_map, args.virtual_stain, args.output_dir)
    print("\n>>> Done.")


if __name__ == "__main__":
    main()
