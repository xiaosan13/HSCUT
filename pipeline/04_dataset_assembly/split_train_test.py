"""
Train/Test Dataset Split
=========================
Paper reference: Section 3.1 Data Preparation

Splits paired A/B images into train/test sets following CycleGAN
directory conventions. Uses random shuffle with fixed seed (42)
for reproducibility.

Output structure:
- trainA/ + trainB/ : Training pairs
- testA/ + testB/   : 120 held-out test pairs

Uses file copy (not move) for data safety.
"""
import os
import sys
import random
import shutil
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pipeline.config_loader import load_config

SRC_A_NAME = 'A'
SRC_B_NAME = 'B'
TEST_COUNT = 120
RANDOM_SEED = 42


def main():
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description='Split paired A/B images into train/test sets')
    parser.add_argument('--data_root',
                        default=os.path.join(cfg['dataset']['root'], 'BF'),
                        help='Root directory containing A/ and B/ subdirectories')
    args = parser.parse_args()

    print("--- Starting Dataset Split (Train/Test) ---")

    src_a = os.path.join(args.data_root, SRC_A_NAME)
    src_b = os.path.join(args.data_root, SRC_B_NAME)

    if not (os.path.exists(src_a) and os.path.exists(src_b)):
        print(f"[Error] Source folders not found: {src_a} or {src_b}")
        return

    train_a_dir = os.path.join(args.data_root, 'trainA')
    train_b_dir = os.path.join(args.data_root, 'trainB')
    test_a_dir = os.path.join(args.data_root, 'testA')
    test_b_dir = os.path.join(args.data_root, 'testB')

    for p in [train_a_dir, train_b_dir, test_a_dir, test_b_dir]:
        os.makedirs(p, exist_ok=True)

    files_a = set(f for f in os.listdir(src_a) if f.endswith('.png'))
    files_b = set(f for f in os.listdir(src_b) if f.endswith('.png'))

    valid_files = list(files_a & files_b)
    total_files = len(valid_files)

    print(f"Found {len(files_a)} in A, {len(files_b)} in B.")
    print(f"Valid Pairs (Intersection): {total_files}")

    if total_files < TEST_COUNT:
        print(f"[Error] Not enough files ({total_files}) to split {TEST_COUNT} for test set.")
        return

    random.seed(RANDOM_SEED)
    random.shuffle(valid_files)

    test_files = valid_files[:TEST_COUNT]
    train_files = valid_files[TEST_COUNT:]

    print(f"Split Result: {len(train_files)} Training, {len(test_files)} Testing.")

    print("Generating Test Set...")
    for f in tqdm(test_files, desc="Copying Test"):
        shutil.copy2(os.path.join(src_a, f), os.path.join(test_a_dir, f))
        shutil.copy2(os.path.join(src_b, f), os.path.join(test_b_dir, f))

    print("Generating Train Set...")
    for f in tqdm(train_files, desc="Copying Train"):
        shutil.copy2(os.path.join(src_a, f), os.path.join(train_a_dir, f))
        shutil.copy2(os.path.join(src_b, f), os.path.join(train_b_dir, f))

    print("\nDataset split completed successfully!")
    print(f"Output Location: {args.data_root}")
    print(f"  - trainA: {len(os.listdir(train_a_dir))} images")
    print(f"  - trainB: {len(os.listdir(train_b_dir))} images")
    print(f"  - testA : {len(os.listdir(test_a_dir))} images")
    print(f"  - testB : {len(os.listdir(test_b_dir))} images")


if __name__ == "__main__":
    main()
