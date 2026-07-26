"""
Cluster Quality Inspection for Background Label Determination
==============================================================
Paper reference: Section 2.2 Upper Branch

Manual inspection tool for trained IPCA+KMeans models.
For pre-selected sample files, generates:
- Cluster assignment maps (visualized in color)
- Per-cluster binary masks (one mask per cluster class)
- Spectral curve plots for each cluster

Purpose: Determine which cluster ID corresponds to background in each
tissue folder (used as --bg_label in CUT model training).

Input:  Trained models, sample HS data files
Output: Cluster maps, spectral curves, per-class masks for manual review
"""
import os
import argparse
import sys
import cv2
import joblib
import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pipeline.config_loader import load_config

PARAMS = {"crop_top": 41, "sigma": 0.5, "n_pca_keep": 20, "n_clusters": 7}

MANUAL_SAMPLES = {
    "CK": ["1-3", "1-6", "6-1"], "CX": ["1-6", "1-9", "3-8"],
    "CY": ["2-4", "3-1", "6-4"], "IK": ["3-1", "5-4", "13-12"],
    "IX": ["4-1", "11-2", "11-3"], "IY": ["2-4", "6-1", "9-3"],
    "JK": ["2-7", "4-1", "4-3"], "JX": ["3-1", "5-6", "6-1"],
    "JY": ["2-4", "2-5", "2-6"], "WK": ["12-1", "14-3", "15-1"],
    "WX": ["10-2", "6-1", "14-1"], "WY": ["5-4", "7-8", "10-3"],
}


def load_models(model_root, folder_name):
    model_dir = os.path.join(model_root, folder_name)
    try:
        scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
        pca = joblib.load(os.path.join(model_dir, "pca.joblib"))
        kmeans = joblib.load(os.path.join(model_dir, "kmeans.joblib"))
        return scaler, pca, kmeans
    except Exception as e:
        print(f"Error loading models for {folder_name}: {e}")
        return None, None, None


def process_data(hdr, spe):
    try:
        img = envi.open(hdr, spe)
        mm = img.open_memmap()
        if mm.shape[0] <= PARAMS["crop_top"]:
            return None
        data = mm[PARAMS["crop_top"]:, :, :300].copy().astype(np.float32)
        if PARAMS["sigma"] > 0:
            for b in range(data.shape[2]):
                data[:, :, b] = gaussian_filter(data[:, :, b], sigma=PARAMS["sigma"])
        return data
    except Exception:
        return None


def plot_spectral_curves(scaler, pca, kmeans, save_path):
    centers_truncated = kmeans.cluster_centers_
    centers_full = np.zeros((centers_truncated.shape[0], pca.n_components_))
    centers_full[:, :centers_truncated.shape[1]] = centers_truncated
    centers_norm = pca.inverse_transform(centers_full)

    plt.figure(figsize=(10, 6))
    for i in range(centers_norm.shape[0]):
        plt.plot(centers_norm[i], label=f'Class {i}')
    plt.title("Average Normalized Spectral Shapes (Destriped)")
    plt.xlabel("Band Index")
    plt.ylabel("Relative Intensity (Normalized)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()


def main():
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description='Inspect IPCA+KMeans cluster quality for background label selection')
    parser.add_argument('--raw_data_root', default=cfg['raw_data']['hs_root'],
                        help='Root directory for raw HS data')
    parser.add_argument('--model_root', default=cfg['models']['ipca_kmeans_dir'],
                        help='Root directory for trained models')
    parser.add_argument('--output_root', default='./output/label_check',
                        help='Output directory for inspection results')
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    print("--- Starting Manual Label Check (L2 Mode) ---")

    for folder, file_ids in MANUAL_SAMPLES.items():
        if not file_ids:
            continue
        print(f"\nProcessing Folder: {folder}...")

        scaler, pca, kmeans = load_models(args.model_root, folder)
        if scaler is None:
            continue

        for fid in file_ids:
            base_path = os.path.join(args.raw_data_root, folder, fid)
            if not os.path.exists(base_path + ".spe"):
                continue

            print(f"  Sample: {fid}")
            sample_out_dir = os.path.join(args.output_root, folder, fid)
            os.makedirs(sample_out_dir, exist_ok=True)

            data = process_data(base_path + ".hdr", base_path + ".spe")
            if data is None:
                continue

            h, w, c = data.shape
            X = data.reshape(-1, 300)
            X_norm = scaler.transform(X)
            X_pca = pca.transform(X_norm)
            X_feats = X_pca[:, :PARAMS["n_pca_keep"]]
            labels = kmeans.predict(X_feats).reshape(h, w)

            # Cluster Map
            plt.figure(figsize=(8, 8))
            plt.imshow(labels, cmap='tab10')
            plt.title(f"Cluster Map: {fid}")
            plt.axis('off')
            plt.savefig(os.path.join(sample_out_dir, "0_Cluster_Map.png"), bbox_inches='tight')
            plt.close()

            # Spectral Curves
            plot_spectral_curves(scaler, pca, kmeans,
                                 os.path.join(sample_out_dir, "0_Spectral_Curves.png"))

            # Per-Class Masks
            for i in range(PARAMS["n_clusters"]):
                mask = np.zeros_like(labels, dtype=np.uint8)
                mask[labels == i] = 255
                mask = cv2.rotate(mask, cv2.ROTATE_180)
                cv2.imwrite(os.path.join(sample_out_dir, f"Mask_Class_{i}.png"), mask)

    print(f"\nDone! Check {args.output_root}")


if __name__ == "__main__":
    main()
