"""
Train IPCA + MiniBatchKMeans Models for Physical Mask Generation
=================================================================
Paper reference: Section 2.2 Upper Branch, Methods Section 3

Trains per-folder dimensionality reduction and clustering models on
raw hyperspectral data for tissue segmentation:

Pipeline (per tissue folder):
1. L2 Normalizer: Destriping via per-pixel vector normalization
2. IncrementalPCA (n_components=300, keep top 20): Spectral dimensionality
   reduction, trained incrementally across all files in folder
3. MiniBatchKMeans (n_clusters=7): Clustering on PCA-reduced features
   with batch_size=480*480 for memory efficiency

Preprocessing per file:
- Crop top 41 rows, extract first 300 bands
- Gaussian smoothing (sigma=0.5) per band

Input:  Raw hyperspectral .spe/.hdr files, whitelist from trainB
Output: scaler.joblib, pca.joblib, kmeans.joblib per folder
"""
import os
import argparse
import sys
import joblib
import numpy as np
import spectral.io.envi as envi
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import Normalizer
from sklearn.cluster import MiniBatchKMeans
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pipeline.config_loader import load_config

PARAMS = {
    "crop_top": 41,
    "sigma": 0.5,
    "n_components": 300,
    "n_pca_keep": 20,
    "n_clusters": 7,
    "batch_size": 480 * 480
}

TARGET_FOLDERS = [
    "CK", "CX", "CY", "IK", "IX", "IY",
    "JK", "JX", "JY", "WK", "WX", "WY"
]


def get_whitelist(whitelist_path, target_folder):
    whitelist = []
    if not os.path.exists(whitelist_path):
        return whitelist
    prefix = target_folder + "_"
    files = [f for f in os.listdir(whitelist_path)
             if f.startswith(prefix) and f.endswith('.png')]
    for f in files:
        whitelist.append(f[len(prefix):-4])
    return whitelist


def load_process_data(hdr, spe):
    try:
        img = envi.open(hdr, spe)
        mm = img.open_memmap()
        if mm.shape[0] <= PARAMS["crop_top"]:
            return None
        data = mm[PARAMS["crop_top"]:, :, :300].copy().astype(np.float32)
        if PARAMS["sigma"] > 0:
            for b in range(data.shape[2]):
                data[:, :, b] = gaussian_filter(data[:, :, b], sigma=PARAMS["sigma"])
        return data.reshape(-1, 300)
    except Exception:
        return None


def train_single_folder(folder_name, file_ids, raw_root, model_root):
    model_dir = os.path.join(model_root, folder_name)
    os.makedirs(model_dir, exist_ok=True)

    scaler = Normalizer(norm='l2')
    ipca = IncrementalPCA(n_components=PARAMS["n_components"])
    kmeans = MiniBatchKMeans(n_clusters=PARAMS["n_clusters"],
                             batch_size=PARAMS["batch_size"], random_state=42)

    valid_paths = []
    for fid in file_ids:
        hdr = os.path.join(raw_root, folder_name, f"{fid}.hdr")
        spe = os.path.join(raw_root, folder_name, f"{fid}.spe")
        if os.path.exists(spe):
            valid_paths.append((fid, hdr, spe))

    if not valid_paths:
        print(f"[{folder_name}] No valid files.")
        return

    print(f"[{folder_name}] 1/3 Scaler (Normalizer is stateless, skipping fit)")

    # Pass 2: PCA
    with tqdm(valid_paths, desc=f"[{folder_name}] 2/3 PCA", leave=False) as pbar:
        for fid, hdr, spe in pbar:
            X = load_process_data(hdr, spe)
            if X is not None:
                ipca.partial_fit(scaler.transform(X))

    # Pass 3: KMeans
    with tqdm(valid_paths, desc=f"[{folder_name}] 3/3 KMeans", leave=False) as pbar:
        for fid, hdr, spe in pbar:
            X = load_process_data(hdr, spe)
            if X is not None:
                X_pca = ipca.transform(scaler.transform(X))
                kmeans.partial_fit(X_pca[:, :PARAMS["n_pca_keep"]])

    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    joblib.dump(ipca, os.path.join(model_dir, "pca.joblib"))
    joblib.dump(kmeans, os.path.join(model_dir, "kmeans.joblib"))

    print(f"[{folder_name}] Finished & Saved")


def main():
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description='Train IPCA + KMeans models for tissue mask generation')
    parser.add_argument('--raw_data_root', default=cfg['raw_data']['hs_root'],
                        help='Root directory for raw HS data')
    parser.add_argument('--whitelist_dir', default=cfg['dataset']['trainA_mask'],
                        help='Directory containing whitelist PNG files')
    parser.add_argument('--model_output_root', default=cfg['models']['ipca_kmeans_dir'],
                        help='Output directory for trained models')
    args = parser.parse_args()

    print("--- Starting Sequential Model Training (L2 Normalized) ---")
    for folder in TARGET_FOLDERS:
        print(f"\nProcessing {folder}...")
        whitelist = get_whitelist(args.whitelist_dir, folder)
        if whitelist:
            train_single_folder(folder, whitelist, args.raw_data_root,
                                args.model_output_root)
        else:
            print(f"[{folder}] Skipped (empty whitelist).")


if __name__ == "__main__":
    main()
