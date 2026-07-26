"""
FID and KID Evaluation Metrics
===============================
Paper reference: Section 2.2 (Figure 2C)

Computes Frechet Inception Distance (FID) and Kernel Inception
Distance (KID) between generated virtual stain images and ground
truth H&E images using the torch-fidelity library.

- FID: Measures macroscopic feature distribution similarity (lower = better)
- KID: Measures local detail distribution similarity (lower = better)

Input:  real_B/ (ground truth) and fake_B/ (generated) directories
Output: Console-printed FID and KID scores
"""
import os
import argparse
import torch
import torch_fidelity
from datetime import datetime

BATCH_SIZE = 4
KID_SUBSET_SIZE = 50


def check_paths(real_dir, fake_dir):
    if not os.path.exists(real_dir):
        print(f"[Error] Real path not found: {real_dir}")
        return False
    if not os.path.exists(fake_dir):
        print(f"[Error] Fake path not found: {fake_dir}")
        return False

    real_files = [f for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    fake_files = [f for f in os.listdir(fake_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Found {len(real_files)} real images.")
    print(f"Found {len(fake_files)} fake images.")

    if len(real_files) == 0 or len(fake_files) == 0:
        print("[Error] One of the folders is empty.")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Compute FID and KID metrics between real and generated images')
    parser.add_argument('--real_dir', required=True,
                        help='Path to real (ground truth) images directory')
    parser.add_argument('--fake_dir', required=True,
                        help='Path to generated (fake) images directory')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size for InceptionV3 inference')
    parser.add_argument('--kid_subset_size', type=int, default=KID_SUBSET_SIZE,
                        help='Subset size for KID computation')
    args = parser.parse_args()

    print("--- Starting Quantitative Analysis (FID & KID) ---")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not check_paths(args.real_dir, args.fake_dir):
        return

    use_cuda = torch.cuda.is_available()
    print(f"Device: {'CUDA' if use_cuda else 'CPU'}")

    print("\nCalculating metrics... (This may take a while to download InceptionV3 model for the first time)")

    try:
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=args.real_dir,
            input2=args.fake_dir,
            cuda=use_cuda,
            batch_size=args.batch_size,
            fid=True,
            kid=True,
            kid_subset_size=args.kid_subset_size,
            verbose=True
        )

        print("\n" + "=" * 40)
        print("FINAL RESULTS")
        print("=" * 40)
        print(f"FID Score (Lower is better): {metrics_dict['frechet_inception_distance']:.4f}")
        print(f"KID Score (Lower is better): {metrics_dict['kernel_inception_distance_mean']:.6f} "
              f"(+/- {metrics_dict['kernel_inception_distance_std']:.6f})")
        print("=" * 40)

        print("\n[Analysis Note]:")
        print("1. FID: Measures distribution distance between generated and real images.")
        print("   - Lower is better. High-quality models typically FID < 50, excellent < 10.")
        print("   - Note: FID may be inflated/unstable with only 120 samples (recommended > 2048).")
        print("2. KID: Similar to FID but more robust for small sample sizes.")
        print("   - Lower is better. More reliable than FID for our 120-sample test set.")

    except Exception as e:
        print(f"\n[Error] Calculation failed: {str(e)}")
        print("Tips: Ensure you have installed torch-fidelity via 'pip install torch-fidelity'")


if __name__ == "__main__":
    main()
