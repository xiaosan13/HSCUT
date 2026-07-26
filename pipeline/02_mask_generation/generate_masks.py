"""
Physical Mask Generation (Batch + Single)
==========================================
Paper reference: Section 2.2 Upper Branch

Applies trained IPCA+KMeans models to generate binary tissue masks.
Supports two modes:
- Batch mode (default): Processes whitelist files for all tissue folders
- Single mode (--single): Processes one specific file for debugging/tuning

Inference pipeline:
1. Read .spe/.hdr, crop top 41 rows, extract first 300 bands
2. Gaussian smoothing (sigma=0.5) per band
3. StandardScaler transform -> PCA transform (keep top 20) -> KMeans predict
4. Extract background cluster (bg_id) as raw binary mask

5-Step Morphological Cleaning:
  Step 1: Majority filter (3x3 kernel, threshold >= 5 neighbors)
  Step 2: Morphological opening (remove white dots)
  Step 3: Morphological closing (connect white gaps)
  Step 4: Remove small white objects (area < MIN_OBJECT_AREA)
  Step 5: Remove small black holes (area < MIN_HOLE_AREA)

Input:  Trained models (scaler, pca, kmeans per folder), raw HS data
Output: Cleaned binary mask PNGs
"""
import os
import argparse
import cv2
import joblib
import numpy as np
import spectral.io.envi as envi
from scipy.ndimage import gaussian_filter
from PIL import Image
from tqdm import tqdm
import warnings
import sys

warnings.filterwarnings('ignore')

# Add project root to path for config_loader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pipeline.config_loader import load_config

BACKGROUND_IDS = {
    "CK": 4, "CX": 4, "CY": 6,
    "IK": 2, "IX": 3, "IY": 0,
    "JK": 3, "JX": 6, "JY": 6,
    "WK": 0, "WX": 1, "WY": 4
}
PARAMS = {"crop_top": 41, "sigma": 0.5, "n_pca_keep": 20}


def get_whitelist(whitelist_dir, target_folder):
    whitelist = []
    if not os.path.exists(whitelist_dir):
        return whitelist
    prefix = target_folder + "_"
    files = [f for f in os.listdir(whitelist_dir)
             if f.startswith(prefix) and f.endswith('.png')]
    for f in files:
        whitelist.append(f[len(prefix):-4])
    return whitelist


def clean_mask_pipeline(binary_img, open_kernel=0, close_kernel=5,
                        min_object_area=300, min_hole_area=10000):
    # Step 1: Majority Filter
    img_norm = (binary_img / 255.0).astype(np.float32)
    kernel = np.ones((3, 3), np.float32)
    neighbor_count = cv2.filter2D(img_norm, -1, kernel, borderType=cv2.BORDER_REFLECT)
    mask_to_kill = neighbor_count < 5
    img = binary_img.copy()
    img[mask_to_kill] = 0

    # Step 2: Opening
    if open_kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, k)

    # Step 3: Closing
    if close_kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)

    # Step 4: Remove Small White Objects
    if min_object_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
        cleaned = np.zeros_like(img)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_object_area:
                cleaned[labels == i] = 255
        img = cleaned

    # Step 5: Remove Small Black Holes
    if min_hole_area > 0:
        inv = cv2.bitwise_not(img)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_hole_area:
                img[labels == i] = 255

    return img


def process_file(raw_root, model_root, folder, fid, bg_id, output_path):
    """Process a single HS file and save mask."""
    hdr = os.path.join(raw_root, folder, f"{fid}.hdr")
    spe = os.path.join(raw_root, folder, f"{fid}.spe")
    if not os.path.exists(spe):
        raise FileNotFoundError(f"File not found: {spe}")

    img = envi.open(hdr, spe)
    mm = img.open_memmap()
    if mm.shape[0] <= PARAMS["crop_top"]:
        raise ValueError("Image too small")
    data = mm[PARAMS["crop_top"]:, :, :300].copy().astype(np.float32)

    if PARAMS["sigma"] > 0:
        for b in range(300):
            data[:, :, b] = gaussian_filter(data[:, :, b], sigma=PARAMS["sigma"])

    h, w = data.shape[:2]
    X = data.reshape(-1, 300)
    X_norm = joblib.load(os.path.join(model_root, folder, "scaler.joblib")).transform(X)
    X_pca = joblib.load(os.path.join(model_root, folder, "pca.joblib")).transform(X_norm)
    X_feats = X_pca[:, :PARAMS["n_pca_keep"]]
    labels = joblib.load(os.path.join(model_root, folder, "kmeans.joblib")).predict(X_feats).reshape(h, w)

    raw = np.zeros_like(labels, dtype=np.uint8)
    raw[labels == bg_id] = 255
    cleaned = clean_mask_pipeline(raw)

    Image.fromarray(cleaned).rotate(180).save(output_path)
    return True


def batch_mode(args):
    os.makedirs(args.output_dir, exist_ok=True)
    total = 0

    for folder, bg_id in BACKGROUND_IDS.items():
        print(f"\nProcessing Folder: {folder} (Background ID: {bg_id})...")
        file_ids = get_whitelist(args.whitelist_dir, folder)
        if not file_ids:
            print("  No files in whitelist, skipping.")
            continue

        model_dir = os.path.join(args.model_root, folder)
        if not os.path.exists(os.path.join(model_dir, "kmeans.joblib")):
            print(f"  Models not found for {folder}, skipping.")
            continue

        folder_count = 0
        for fid in tqdm(file_ids, desc=f"  Generating"):
            try:
                output_path = os.path.join(args.output_dir, f"{folder}_{fid}.png")
                process_file(args.raw_data_root, args.model_root, folder, fid, bg_id, output_path)
                folder_count += 1
            except Exception:
                pass

        print(f"  Finished {folder}: {folder_count} masks.")
        total += folder_count

    print(f"\nAll Done! Total masks: {total}")


def single_mode(args):
    output_path = args.output or f"mask_{args.folder}_{args.file}.png"
    try:
        process_file(args.raw_data_root, args.model_root,
                     args.folder, args.file, args.bg_id, output_path)
        print(f"Saved mask to: {output_path}")
    except Exception as e:
        print(f"Error: {e}")


def main():
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description='Generate binary tissue masks using IPCA+KMeans models')
    parser.add_argument('--raw_data_root', default=cfg['raw_data']['hs_root'],
                        help='Root directory for raw HS data')
    parser.add_argument('--model_root', default=cfg['models']['ipca_kmeans_dir'],
                        help='Root directory for trained IPCA+KMeans models')
    parser.add_argument('--output_dir', default=cfg['processed']['masks_root'],
                        help='Output directory for generated masks')

    # Batch mode options
    parser.add_argument('--whitelist_dir', default=cfg['dataset']['trainA_mask'],
                        help='Whitelist directory for file IDs (batch mode)')

    # Single mode options
    parser.add_argument('--single', action='store_true',
                        help='Process a single file instead of batch')
    parser.add_argument('--folder', help='Tissue folder code (e.g. CK, IK)')
    parser.add_argument('--file', help='File ID without extension')
    parser.add_argument('--bg_id', type=int, help='Background cluster ID for this folder')
    parser.add_argument('--output', help='Output path for single file mask')

    args = parser.parse_args()

    if args.single:
        if not all([args.folder, args.file, args.bg_id]):
            parser.error('--single requires --folder, --file, and --bg_id')
        single_mode(args)
    else:
        batch_mode(args)


if __name__ == "__main__":
    main()
