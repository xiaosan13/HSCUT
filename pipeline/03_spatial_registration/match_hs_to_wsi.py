"""
Hyperspectral-to-WSI Spatial Registration (Strict + Rescue Modes)
==================================================================
Paper reference: Section 3.1 Data Preparation

Spatial registration pipeline using SIFT feature matching with two modes:
- strict (default): Tight thresholds for primary matching
- rescue: Relaxed thresholds with timeout to capture missed samples

Pipeline:
1. SIFT feature detection + FLANN matching between HS TrueColor and BF sub-image
2. Homography estimation with RANSAC
3. SSIM (Structural Similarity) refinement via sliding window for precise localization
4. Histogram matching for color consistency
5. Size filtering
6. Crops matching regions from BF, TS (H&E WSI), and HS for triplet dataset

Input:  Raw HS data, BF sub-images (BF_Subimages_Filtered/), TS.png WSI
Output: Triplet images: BF/A, BF/B, HS/B
"""
import os
import argparse
import sys
import cv2
import numpy as np
import glob
import spectral.io.envi as envi
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import time
from PIL import Image
import gc

Image.MAX_IMAGE_PIXELS = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pipeline.config_loader import load_config

CROP_TOP = 41
RGB_BANDS = [122, 62, 33]
NORM_PERCENTILE = (2, 98)
SUB_DIR_NAME = 'BF_Subimages_Filtered'

# Matching parameter presets
STRICT_PARAMS = {
    'min_match_count': 7, 'ratio_thresh': 0.72, 'ransac_thresh': 6.0,
    'min_size': 480, 'max_size': 550
}
RESCUE_PARAMS = {
    'min_match_count': 5, 'ratio_thresh': 0.8, 'ransac_thresh': 8.0,
    'min_size': 480, 'max_size': 650
}


def load_reference_data(folder_path):
    ref_hdr = os.path.join(folder_path, '0.hdr')
    ref_spe = os.path.join(folder_path, '0.spe')
    if not (os.path.exists(ref_hdr) and os.path.exists(ref_spe)):
        return None
    try:
        ref_obj = envi.open(ref_hdr, ref_spe)
        ref_mem = ref_obj.open_memmap()
        if ref_mem.shape[0] <= CROP_TOP:
            return None
        return ref_mem[CROP_TOP:, :, :300].astype(np.float32)
    except Exception:
        return None


def process_raw_spe(hdr_path, spe_path, ref_data):
    try:
        img_obj = envi.open(hdr_path, spe_path)
        raw_mem = img_obj.open_memmap()
        if raw_mem.shape[0] <= CROP_TOP:
            return None, None
        data_subset = raw_mem[CROP_TOP:, :, :300].astype(np.float32)
        data_cal = data_subset / (ref_data + 1.0) if ref_data is not None else data_subset
        data_rot = cv2.rotate(data_cal, cv2.ROTATE_180)
        rgb_data = data_rot[:, :, RGB_BANDS]
        img_rgb = np.zeros_like(rgb_data, dtype=np.uint8)
        for i in range(3):
            band = rgb_data[:, :, i]
            p_min, p_max = np.percentile(band, NORM_PERCENTILE)
            if p_max > p_min:
                norm = (band - p_min) / (p_max - p_min) * 255.0
                img_rgb[:, :, i] = np.clip(norm, 0, 255).astype(np.uint8)
        return img_rgb, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    except Exception:
        return None, None


def match_histograms(source, reference):
    src_hist, _ = np.histogram(source.flatten(), 256, [0, 256])
    ref_hist, _ = np.histogram(reference.flatten(), 256, [0, 256])
    src_cdf = src_hist.cumsum()
    ref_cdf = ref_hist.cumsum()
    src_cdf_norm = (src_cdf - src_cdf.min()) * 255 / ((src_cdf.max() - src_cdf.min()) + 1e-5)
    ref_cdf_norm = (ref_cdf - ref_cdf.min()) * 255 / ((ref_cdf.max() - ref_cdf.min()) + 1e-5)
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lut[i] = np.argmin(np.abs(src_cdf_norm[i] - ref_cdf_norm))
    return cv2.LUT(source, lut)


def calculate_ssim_sliding_window(template, search_region):
    t_h, t_w = template.shape
    r_h, r_w = search_region.shape
    best_score = -1.0
    best_x, best_y = 0, 0
    y_steps = r_h - t_h + 1
    x_steps = r_w - t_w + 1
    if y_steps <= 0 or x_steps <= 0:
        return 0, 0, 0.0
    for y in range(y_steps):
        for x in range(x_steps):
            score = ssim(template, search_region[y:y + t_h, x:x + t_w], data_range=255)
            if score > best_score:
                best_score = score
                best_x = x
                best_y = y
    return best_x, best_y, best_score


def find_location_in_subimage(zc_img, bf_sub, sub_y, sub_x, params, expand=5):
    zc_matched = match_histograms(zc_img, bf_sub)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(zc_matched, None)
    kp2, des2 = sift.detectAndCompute(bf_sub, None)
    if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
        return None

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except Exception:
        return None

    good = [m for m, n in matches if m.distance < params['ratio_thresh'] * n.distance]
    if len(good) < params['min_match_count']:
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, params['ransac_thresh'])
    if M is None:
        return None

    h, w = zc_img.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    x_min = int(np.min(dst[:, 0, 0]))
    x_max = int(np.max(dst[:, 0, 0]))
    y_min = int(np.min(dst[:, 0, 1]))
    y_max = int(np.max(dst[:, 0, 1]))
    est_w = x_max - x_min
    est_h = y_max - y_min

    if est_w < params['min_size'] or est_h < params['min_size']:
        return None
    if est_w > params['max_size'] or est_h > params['max_size']:
        return None

    search_x1 = max(0, x_min - expand)
    search_y1 = max(0, y_min - expand)
    search_x2 = min(bf_sub.shape[1], x_max + expand)
    search_y2 = min(bf_sub.shape[0], y_max + expand)
    search_roi = bf_sub[search_y1:search_y2, search_x1:search_x2]
    if search_roi.shape[0] < h or search_roi.shape[1] < w:
        return None

    lx, ly, score = calculate_ssim_sliding_window(zc_matched, search_roi)
    return (sub_x + search_x1 + lx, sub_y + search_y1 + ly, score)


def process_folder(folder_name, args, params):
    raw_folder = os.path.join(args.raw_data_root, folder_name)
    bf_sub_dir = os.path.join(args.gt_root, folder_name, 'BF_Subimages_Filtered')
    ts_path = os.path.join(args.gt_root, folder_name, 'TS.png')

    if not os.path.exists(raw_folder) or not os.path.exists(bf_sub_dir):
        print(f"Skipping {folder_name}: Data missing.")
        return 0

    ref_data = load_reference_data(raw_folder)
    print(f"Loading TS image: {ts_path}")
    try:
        ts_img_large = Image.open(ts_path)
        ts_w, ts_h = ts_img_large.size
    except Exception as e:
        print(f"Failed to load TS image: {e}")
        return 0

    bf_subs = []
    for f in glob.glob(os.path.join(bf_sub_dir, "*.png")):
        try:
            base = os.path.splitext(os.path.basename(f))[0]
            parts = base.split('_')
            y_start, x_start = int(parts[1]), int(parts[2])
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                bf_subs.append({'img': img, 'y': y_start, 'x': x_start})
        except Exception:
            pass

    spe_files = glob.glob(os.path.join(raw_folder, "*.spe"))
    spe_files = [f for f in spe_files if not f.endswith('0.spe')]
    print(f"Matching {len(spe_files)} raw SPE files...")
    match_count = 0

    for spe_path in tqdm(spe_files, desc=f"{args.mode.capitalize()} {folder_name}"):
        file_id = os.path.splitext(os.path.basename(spe_path))[0]
        save_name = f"{folder_name}_{file_id}.png"

        # Skip check
        if os.path.exists(os.path.join(args.out_hs_b, save_name)):
            continue

        # Rescue mode: also skip if already in FIN-DATA
        if args.mode == 'rescue':
            if args.fin_data_dir and os.path.exists(
                    os.path.join(args.fin_data_dir, 'HS', 'B', save_name)):
                continue

        hdr_path = os.path.splitext(spe_path)[0] + ".hdr"
        if not os.path.exists(hdr_path):
            continue

        hs_rgb, hs_gray = process_raw_spe(hdr_path, spe_path, ref_data)
        if hs_gray is None:
            continue

        target_h, target_w = hs_gray.shape
        best_match = None
        best_score = -1.0
        start_time = time.time()

        for sub_data in bf_subs:
            if sub_data['img'].shape[0] < target_h or sub_data['img'].shape[1] < target_w:
                continue
            res = find_location_in_subimage(hs_gray, sub_data['img'],
                                            sub_data['y'], sub_data['x'], params)
            if res:
                gx, gy, score = res
                if score > best_score:
                    best_score = score
                    best_match = {'gx': gx, 'gy': gy, 'sub': sub_data}

            if args.mode == 'rescue' and args.timeout and \
                    time.time() - start_time > args.timeout:
                break

        if best_match:
            gx, gy = best_match['gx'], best_match['gy']
            sub_data = best_match['sub']
            local_x = gx - sub_data['x']
            local_y = gy - sub_data['y']
            sub_h, sub_w = sub_data['img'].shape

            valid_ts = (gx + target_w <= ts_w) and (gy + target_h <= ts_h)
            valid_bf = (local_x >= 0 and local_y >= 0 and
                        local_x + target_w <= sub_w and local_y + target_h <= sub_h)

            if valid_ts and valid_bf:
                try:
                    ts_crop_pil = ts_img_large.crop((gx, gy, gx + target_w, gy + target_h))
                    ts_crop = cv2.cvtColor(np.array(ts_crop_pil), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(args.out_bf_b, save_name), ts_crop)
                except Exception as e:
                    print(f"Error saving TS crop: {e}")
                    continue

                bf_crop = sub_data['img'][local_y:local_y + target_h,
                                          local_x:local_x + target_w]
                cv2.imwrite(os.path.join(args.out_bf_a, save_name), bf_crop)
                cv2.imwrite(os.path.join(args.out_hs_b, save_name),
                            cv2.cvtColor(hs_rgb, cv2.COLOR_RGB2BGR))
                match_count += 1

    print(f"Finished {folder_name}: {match_count} matched.")
    del bf_subs
    gc.collect()
    return match_count


def main():
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description='HS-to-WSI spatial registration with SIFT+SSIM')
    parser.add_argument('--raw_data_root', default=cfg['raw_data']['hs_root'],
                        help='Root directory for raw HS data')
    parser.add_argument('--gt_root', default=cfg['raw_data']['wsi_root'],
                        help='Root directory for ground truth WSI data')
    parser.add_argument('--output_root', default=cfg['processed']['registered_root'],
                        help='Output root for registered triplets')
    parser.add_argument('--mode', choices=['strict', 'rescue'], default='strict',
                        help='Matching mode: strict (tight) or rescue (loose)')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Per-file timeout in seconds (rescue mode default: 360)')
    parser.add_argument('--fin_data_dir', default=None,
                        help='FIN-DATA directory to skip already-matched files (rescue mode)')
    parser.add_argument('--folders', nargs='+',
                        default=['1514619-A7', '1577035-C4', '1816663-A8'],
                        help='Folder names to process')
    args = parser.parse_args()

    params = RESCUE_PARAMS if args.mode == 'rescue' else STRICT_PARAMS

    if args.mode == 'rescue':
        if args.timeout is None:
            args.timeout = 360
        if args.fin_data_dir is None:
            args.fin_data_dir = cfg['processed']['registered_root']
        out_root = os.path.join(args.output_root, '..', 'TEMP-DATA')
    else:
        out_root = args.output_root

    args.out_bf_a = os.path.join(out_root, 'BF', 'A')
    args.out_bf_b = os.path.join(out_root, 'BF', 'B')
    args.out_hs_b = os.path.join(out_root, 'HS', 'B')
    for p in [args.out_bf_a, args.out_bf_b, args.out_hs_b]:
        os.makedirs(p, exist_ok=True)

    print(f"--- Starting {args.mode.upper()} Mode Matching ---")

    for folder_name in args.folders:
        print(f"\n{'=' * 60}")
        process_folder(folder_name, args, params)

    print("\nAll tasks completed.")


if __name__ == "__main__":
    main()
