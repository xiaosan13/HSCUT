"""
Sliding-Window Tile Cropping
=============================
Paper reference: Section 3.1 Data Preparation

Simple sliding-window crop of WSI image into fixed-size tiles
with configurable overlap. Used as a fast alternative when
SIFT-based matching is not needed.

Input:  Target image (PNG/TIFF)
Output: Cropped tiles
"""
import os
import argparse
import cv2
from tqdm import tqdm

CROP_SIZE = 480
OVERLAP_RATE = 0.2


def crop_slide_window(img_path, save_dir, crop_size, overlap_rate):
    print(f"Loading image from: {img_path}")
    img = cv2.imread(img_path)

    if img is None:
        print("Error: Could not load image. Please check the path.")
        return

    h, w = img.shape[:2]
    print(f"Original Image Size: {w} x {h}")

    overlap_px = int(crop_size * overlap_rate)
    stride = crop_size - overlap_px
    print(f"Crop Size: {crop_size}x{crop_size}, Overlap: {overlap_px} px, Stride: {stride} px")

    os.makedirs(save_dir, exist_ok=True)
    print(f"Created output directory: {save_dir}")

    coords = []
    for y in range(0, h - crop_size + 1, stride):
        for x in range(0, w - crop_size + 1, stride):
            coords.append((x, y))

    print(f"Total tiles to generate: {len(coords)}")
    count = 0

    for x, y in tqdm(coords, desc="Cropping"):
        crop_img = img[y:y + crop_size, x:x + crop_size]
        if crop_img.shape[0] != crop_size or crop_img.shape[1] != crop_size:
            continue
        save_path = os.path.join(save_dir, f"tile_y{y}_x{x}.png")
        cv2.imwrite(save_path, crop_img)
        count += 1

    print(f"\nProcessing complete. Saved {count} tiles to {save_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Sliding-window tile cropping from a WSI image')
    parser.add_argument('--input', required=True,
                        help='Path to input image file')
    parser.add_argument('--output_dir', required=True,
                        help='Directory to save cropped tiles')
    parser.add_argument('--crop_size', type=int, default=480,
                        help='Size of square crop (default: 480)')
    parser.add_argument('--overlap', type=float, default=0.2,
                        help='Overlap rate (default: 0.2)')
    args = parser.parse_args()

    crop_slide_window(args.input, args.output_dir, args.crop_size, args.overlap)


if __name__ == "__main__":
    main()
