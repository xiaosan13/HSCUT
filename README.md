# HSCUT: Hyperspectral Contrastive Unpaired Translation

Code repository for the paper **"HSCUT Network Based on Hyperspectral Features and Mask Constraints"**.

HSCUT is a physics-informed unsupervised deep learning framework that directly maps high-dimensional hyperspectral microscopy images to high-fidelity virtual H&E-stained images, without requiring paired training data.

## Setup

```bash
# Clone the repository
git clone https://github.com/your-org/HSCUT.git
cd HSCUT

# Install dependencies
pip install -r requirements.txt

# Configure data paths
cp configs/paths.yaml.example configs/paths.yaml
# Edit configs/paths.yaml to match your local data directory structure
```

## Data Preparation

Expected directory structure for `paths.yaml`:

```
<your_data_root>/
├── HS-DATA/                    # Raw .spe/.hdr hyperspectral files
│   ├── CK/                     # One subfolder per tissue sample
│   │   ├── 0.hdr / 0.spe       # Reference file (white reference)
│   │   ├── 1-3.hdr / 1-3.spe   # Measurement files
│   │   └── ...
│   ├── IK/
│   └── ...
├── GT-DATA/                    # Whole-slide OME-TIFF images
│   ├── 1514619-A7/
│   │   ├── BF.png              # Bright-field sub-image
│   │   └── TS.png              # True-color whole-slide image
│   └── ...
├── FIN-DATA/                   # Final dataset for CUT training
│   ├── BF/
│   │   ├── A/  B/              # Paired BF images
│   │   ├── trainA/ trainB/     # Training split
│   │   └── testA/  testB/      # Test split (120 pairs)
│   └── HS/
│       ├── trainA/ trainB/
│       ├── testA/  testB/
│       └── trainA_mask/        # Binary masks (IPCA+KMeans output)
└── Models_Separate_l2/         # Trained IPCA + KMeans models
    ├── CK/  (scaler.joblib, pca.joblib, kmeans.joblib)
    └── ...
```

## Project Structure

```
HSCUT/
├── pipeline/                          # Full data processing pipeline
│   ├── config_loader.py               # YAML config autoloader
│   ├── 01_preprocessing/              # Feature extraction (TrueColor + FFT/MNF)
│   ├── 02_mask_generation/            # Physical mask via IPCA + K-Means
│   ├── 03_spatial_registration/       # HS-to-WSI matching & channel merging
│   ├── 04_dataset_assembly/           # Train/test split & cleanup
│   └── 05_physical_decoupling/        # Cauchy+LM fitting & validation
├── models/                            # CUT model with mask loss (NetG/D/F)
├── data/                              # Dataset loaders (with mask support)
├── options/                           # Command-line argument parsing
├── util/                              # Visualization, HTML, utilities
├── evaluate/                          # FID/KID metrics
├── configs/                           # Path configuration (paths.yaml.example)
├── train.py                           # Main training entry point
├── test.py                            # Inference entry point
├── train_kmeans.py                    # K-Means training for mask loss
└── requirements.txt                   # Python dependencies
```

## Pipeline Overview

```
Raw HS Cube (480x480x300)
    │
    ├──> [01_preprocessing]  TrueColor + FFT features
    ├──> [02_mask_generation] IPCA + KMeans -> binary tissue mask
    ├──> [03_registration]    SIFT matching + channel merging
    ├──> [04_assembly]        trainA/B + testA/B splits
    │
    ▼
[NetG] ──── virtual H&E image
    │
    ├── loss_GAN  (NetD):  adversarial realism
    ├── loss_NCE  (NetF):  contrastive structure preservation
    └── loss_MASK (K-Means): one-way background inclusion
```

## Key Components

### Physical Mask Generation (Upper Branch)
- **IPCA** (Incremental PCA): Reduces 300 spectral bands to 20 principal components
- **MiniBatchKMeans**: Clusters PCA features into tissue/background classes
- **5-Step Morphological Cleaning**: Majority filter, open, close, area filtering, hole filling

### Feature Extraction (Lower Branch)
- **TrueColor**: Synthesized from spectral bands R122, G62, B33
- **FFT Components**: Fourier transform along spectral axis extracts frequency-domain features
- **Channel Merging**: FFT + TrueColor fused into 3-channel network input

### Physical Decoupling (Calculation-Map)
- **Cauchy Dispersion Model**: Delta_n(lambda) = A + B/lambda^2
- **Interference + Scattering Model**: I(lambda) = envelope * interference + diffuse
- **Levenberg-Marquardt Fitting**: Per-pixel nonlinear optimization extracts h_avg and sigma_h

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure paths in `configs/paths.yaml`.

3. Run the pipeline sequentially:
   ```bash
   # Step 1: Extract features from raw HS data
   python pipeline/01_preprocessing/extract_features.py --input_dir <raw_data> --output_dir <features>
   python pipeline/01_preprocessing/downsample_wsi.py --input_dir <wsi_data> --output_dir <downsampled>

   # Step 2: Generate physical masks
   python pipeline/02_mask_generation/train_ipca_kmeans.py
   python pipeline/02_mask_generation/generate_masks.py

   # Step 3: Spatial registration
   python pipeline/03_spatial_registration/match_hs_to_wsi.py
   python pipeline/03_spatial_registration/merge_channels.py --output_dir <merged>

   # Step 4: Assemble dataset
   python pipeline/04_dataset_assembly/split_train_test.py --data_root <dataset>

   # Step 5: Train K-Means for mask loss
   python train_kmeans.py --input_dir <path_to_trainB>

   # Step 6: Train HSCUT
   python train.py --dataroot <dataset_root> --name hscut_experiment \
       --CUT_mode CUT --lambda_mask 1.0 --kmeans_path ./k_means/kmeans_model.pkl

   # Step 7: Test
   python test.py --dataroot <dataset_root> --name hscut_experiment
   ```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code, please cite our paper.
