"""
Whole-Slide Image Downsampling
===============================
Paper reference: Section 3.1 Data Preparation

Downsamples OME-TIFF whole-slide images from original resolution
(0.274 um/px) to target resolution (1.2 um/px) to match hyperspectral
spatial resolution. Uses INTER_AREA interpolation for anti-aliased
downscaling.

Output: BF.png (grayscale) and TS.png (RGB) for spatial registration.
"""
import os
import argparse
import cv2
import tifffile
import numpy as np

ORIGIN_RES = 0.274
TARGET_RES = 1.2


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created output directory: {directory}")


def process_and_save(input_dir, output_dir, input_filename, output_filename,
                     is_grayscale=False):
    input_path = os.path.join(input_dir, input_filename)
    output_path = os.path.join(output_dir, output_filename)

    print(f"Processing: {input_filename} ...")

    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        return

    try:
        img = tifffile.imread(input_path)
    except Exception as e:
        print(f"Read error: {e}")
        return

    print(f"  - Original size: {img.shape}")

    if img.ndim == 3 and img.shape[0] < 10 and img.shape[2] > 100:
        img = np.transpose(img, (1, 2, 0))
    elif img.ndim > 3:
        img = img[0]
        if img.ndim == 3 and img.shape[0] < 10:
            img = np.transpose(img, (1, 2, 0))

    scale_factor = ORIGIN_RES / TARGET_RES
    new_width = int(img.shape[1] * scale_factor)
    new_height = int(img.shape[0] * scale_factor)

    print(f"  - Scale factor: {scale_factor:.4f}")
    print(f"  - Target size: ({new_height}, {new_width})")

    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    if is_grayscale:
        if len(resized_img.shape) == 3:
            gray = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = resized_img
        cv2.imwrite(output_path, gray)
        print(f"  - Saved grayscale: {output_path}")
    else:
        if len(resized_img.shape) == 3:
            bgr_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR)
        elif len(resized_img.shape) == 2:
            bgr_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)
        else:
            bgr_img = resized_img
        cv2.imwrite(output_path, bgr_img)
        print(f"  - Saved RGB: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Downsample OME-TIFF WSI to match HS resolution')
    parser.add_argument('--input_dir', required=True,
                        help='Directory containing registered OME-TIFF slides')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for BF.png and TS.png')
    parser.add_argument('--file_gray', required=True,
                        help='OME-TIFF filename for grayscale output (BF.png)')
    parser.add_argument('--file_rgb', required=True,
                        help='OME-TIFF filename for RGB output (TS.png)')
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    process_and_save(args.input_dir, args.output_dir, args.file_gray,
                     "BF.png", is_grayscale=True)
    print("-" * 30)
    process_and_save(args.input_dir, args.output_dir, args.file_rgb,
                     "TS.png", is_grayscale=False)

    print("\nAll tasks complete.")


if __name__ == "__main__":
    main()
