"""
Multi-Channel Feature Fusion: FFT + TrueColor Merging
======================================================
Paper reference: Section 2.2 Lower Branch

Merges FFT component images (FFT0, FFT1) and TrueColor images into
3-channel pseudo-RGB images for network input.

Channel mapping: [FFT1, FFT0, TC] -> [Blue, Green, Red]

Processing:
- Optional whitelist filtering: only process files present in target domain B
- Column destriping: global gain correction computed from column means
  to remove vertical stripe artifacts

Input:  FFT0/, FFT1/, True_color/ per folder; optional whitelist from HS/B/
Output: Destriped 3-channel images (network training input)
"""
import os
import argparse
import sys
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pipeline.config_loader import load_config


def get_whitelist_set(whitelist_path):
    if not whitelist_path or not os.path.exists(whitelist_path):
        return None
    files = set(f for f in os.listdir(whitelist_path) if f.endswith('.png'))
    print(f"Loaded Whitelist: {len(files)} files from {whitelist_path}")
    return files


def load_and_merge_data(input_root, folders, whitelist_set):
    merged_images = []
    output_filenames = []

    print(f"Scanning {len(folders)} folders in {input_root}...")

    for folder in folders:
        dir_tc = os.path.join(input_root, folder, 'True_color')
        dir_fft0 = os.path.join(input_root, folder, 'FFT0')
        dir_fft1 = os.path.join(input_root, folder, 'FFT1')

        if not all(os.path.exists(d) for d in [dir_tc, dir_fft0, dir_fft1]):
            continue

        all_files = [f for f in os.listdir(dir_tc) if f.endswith('.png')]
        files_to_process = all_files
        if whitelist_set is not None:
            files_to_process = [f for f in all_files if f in whitelist_set]

        if not files_to_process:
            continue

        for fname in tqdm(files_to_process, desc=f"Merging {folder}"):
            path_tc = os.path.join(dir_tc, fname)
            path_fft0 = os.path.join(dir_fft0, fname)
            path_fft1 = os.path.join(dir_fft1, fname)

            if os.path.exists(path_fft0) and os.path.exists(path_fft1):
                img_tc = cv2.imread(path_tc, cv2.IMREAD_GRAYSCALE)
                img_fft0 = cv2.imread(path_fft0, cv2.IMREAD_GRAYSCALE)
                img_fft1 = cv2.imread(path_fft1, cv2.IMREAD_GRAYSCALE)

                if img_tc is None or img_fft0 is None or img_fft1 is None:
                    continue

                merged = cv2.merge([img_fft1, img_fft0, img_tc])
                merged_images.append(merged)
                output_filenames.append(fname)

    if not merged_images:
        print("No images found/merged. Check your paths or whitelist.")
        return None, None

    print("Stacking images into memory...")
    data_stack = np.array(merged_images, dtype=np.float32)
    print(f"Total dataset shape: {data_stack.shape}")
    return data_stack, output_filenames


def calculate_destriping_gain(image_stack):
    print("Calculating global destriping statistics...")
    col_means = np.mean(image_stack, axis=(0, 1)) + 1e-6
    global_means = np.mean(col_means, axis=0)
    print(f"Global Channel Means (BGR): {global_means}")
    gain = global_means / col_means
    return gain[np.newaxis, np.newaxis, :, :]


def apply_and_save(image_stack, gain_matrix, filenames, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Applying correction and saving to: {output_dir}")
    corrected = np.clip(image_stack * gain_matrix, 0, 255).astype(np.uint8)
    for i, fname in enumerate(tqdm(filenames, desc="Saving Images")):
        cv2.imwrite(os.path.join(output_dir, fname), corrected[i])


def main():
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description='Merge FFT + TrueColor channels into 3-channel images')
    parser.add_argument('--input_root', default=cfg['raw_data']['hs_root'],
                        help='Root directory containing FFT/True_color subdirectories')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for merged images')
    parser.add_argument('--whitelist_dir', default=None,
                        help='Optional whitelist directory (only merge files present here)')
    parser.add_argument('--folders', nargs='+',
                        default=['1514619-A7', '1577035-C4'],
                        help='Folder names to process')
    args = parser.parse_args()

    print("--- Starting Final Merge & Destripe Process ---")

    whitelist_set = get_whitelist_set(args.whitelist_dir)

    data_stack, filenames = load_and_merge_data(args.input_root, args.folders, whitelist_set)

    if data_stack is not None:
        gain = calculate_destriping_gain(data_stack)
        apply_and_save(data_stack, gain, filenames, args.output_dir)
        print("\nAll tasks completed successfully.")
        print(f"Final output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
