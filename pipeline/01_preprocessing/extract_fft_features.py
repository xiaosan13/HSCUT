"""
Feature Extraction (Pure FFT): TrueColor + FFT Components
==========================================================
Paper reference: Section 2.2 Lower Branch

Parallel feature extraction producing individual FFT magnitude components.
Generates from raw hyperspectral data:
1. TrueColor images: RGB from specified bands with percentile normalization
2. FFT components: FFT0, FFT1, FFT2 magnitude images along spectral axis

Processing: crop top 41 rows -> calibrate against reference (0.spe) ->
rotate 180 deg -> TrueColor + FFT extraction in parallel via ProcessPoolExecutor

Input:  Raw .spe/.hdr files
Output: True_color/, FFT0/, FFT1/, FFT2/ subdirectories per folder
"""
import os
import argparse
import numpy as np
import spectral.io.envi as envi
from PIL import Image
import cv2
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import multiprocessing

CROP_TOP = 41
RGB_BANDS = [122, 62, 33]
NORM_PERCENTILE = (2, 98)


def normalize_minmax(data):
    v_min, v_max = np.min(data), np.max(data)
    if v_max - v_min > 1e-6:
        norm = (data - v_min) / (v_max - v_min) * 255.0
    else:
        norm = np.zeros_like(data)
    return norm.astype(np.uint8)


def process_single_folder(args_pack):
    folder_name, pos, input_root, output_root, target_folders = args_pack
    folder_path = os.path.join(input_root, folder_name)

    sub_dirs = {
        'FFT0': os.path.join(output_root, folder_name, 'FFT0'),
        'FFT1': os.path.join(output_root, folder_name, 'FFT1'),
        'FFT2': os.path.join(output_root, folder_name, 'FFT2'),
        'True_color': os.path.join(output_root, folder_name, 'True_color')
    }
    for p in sub_dirs.values():
        os.makedirs(p, exist_ok=True)

    ref_hdr = os.path.join(folder_path, '0.hdr')
    ref_spe = os.path.join(folder_path, '0.spe')

    if not (os.path.exists(ref_hdr) and os.path.exists(ref_spe)):
        return f"Skipped {folder_name}: Missing reference files (0.hdr/0.spe)."

    try:
        ref_obj = envi.open(ref_hdr, ref_spe)
        ref_data = ref_obj.open_memmap()

        if ref_data.shape[0] <= CROP_TOP:
            return f"Error {folder_name}: Reference image height too small."

        ref_subset = ref_data[CROP_TOP:, :, :300].astype(np.float32)

        all_hdr_files = [f for f in os.listdir(folder_path) if f.endswith('.hdr') and f != '0.hdr']
        target_files = [os.path.splitext(f)[0] for f in all_hdr_files]

        if not target_files:
            return f"Skipped {folder_name}: No data files found."

        for file_id in tqdm(target_files, desc=f"Worker {pos}: {folder_name}",
                            position=pos + 1, leave=False):
            input_hdr = os.path.join(folder_path, f"{file_id}.hdr")
            input_spe = os.path.join(folder_path, f"{file_id}.spe")

            if not os.path.exists(input_spe):
                continue

            img_obj = envi.open(input_hdr, input_spe)
            raw_data = img_obj.open_memmap()
            if raw_data.shape[0] <= CROP_TOP:
                continue

            data_subset = raw_data[CROP_TOP:, :, :300].astype(np.float32)
            data_cal = data_subset / (ref_subset + 1.0)
            data_rot = cv2.rotate(data_cal, cv2.ROTATE_180)

            # True Color
            rgb_data = data_rot[:, :, RGB_BANDS]
            rgb_final = np.zeros_like(rgb_data, dtype=np.uint8)
            for i in range(3):
                band = rgb_data[:, :, i]
                p_min, p_max = np.percentile(band, NORM_PERCENTILE)
                if p_max > p_min:
                    band_norm = (band - p_min) / (p_max - p_min) * 255.0
                    rgb_final[:, :, i] = np.clip(band_norm, 0, 255).astype(np.uint8)

            save_name = f"{folder_name}_{file_id}.png"
            Image.fromarray(rgb_final).save(os.path.join(sub_dirs['True_color'], save_name))

            # FFT Components
            fft_all = np.fft.fft(data_rot, axis=2)
            for i in range(3):
                fft_mag = np.abs(fft_all[:, :, i])
                fft_img = normalize_minmax(fft_mag)
                Image.fromarray(fft_img).save(
                    os.path.join(sub_dirs[f'FFT{i}'], f"{folder_name}_{file_id}.png"))

    except Exception as e:
        return f"Error in {folder_name}: {str(e)}"

    return f"Finished: {folder_name}"


def main():
    parser = argparse.ArgumentParser(
        description='Extract TrueColor + individual FFT components from raw HS data')
    parser.add_argument('--input_dir', required=True,
                        help='Root directory of raw hyperspectral data')
    parser.add_argument('--output_dir', required=True,
                        help='Root directory for output')
    parser.add_argument('--target_folders', nargs='*', default=[],
                        help='Specific folder names to process (default: all folders)')
    parser.add_argument('--max_workers', type=int, default=1,
                        help='Maximum number of parallel workers')
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return

    all_folders = [d for d in os.listdir(args.input_dir)
                   if os.path.isdir(os.path.join(args.input_dir, d))]
    all_folders.sort()

    if args.target_folders:
        folders_to_process = [f for f in all_folders if f in args.target_folders]
        print(f"Target filter active. Processing: {folders_to_process}")
        missing = set(args.target_folders) - set(all_folders)
        if missing:
            print(f"Warning: Folders not found: {missing}")
        if not folders_to_process:
            print("No valid folders to process. Exiting.")
            return
    else:
        folders_to_process = all_folders
        print(f"Processing all {len(all_folders)} folders.")

    max_workers = min(args.max_workers, len(folders_to_process))
    task_args = [(folder, i % max_workers, args.input_dir, args.output_dir,
                  args.target_folders)
                 for i, folder in enumerate(folders_to_process)]

    print(f"Starting with {max_workers} workers. Output: {args.output_dir}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(process_single_folder, task_args),
                  total=len(task_args), desc="Total Progress", position=0))

    print("\nAll tasks completed successfully.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
