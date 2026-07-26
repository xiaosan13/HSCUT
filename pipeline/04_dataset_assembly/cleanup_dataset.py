"""
Dataset Cleanup: Remove Bad/Error-Listed Samples
==================================================
Deletes specified files from multiple subdirectories under a
dataset root. Supports two modes:
- --error-file: Read sample IDs from an error list file
- --ids: Provide sample IDs directly on the command line

Includes a 3-second safety delay and post-deletion verification.
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pipeline.config_loader import load_config


def main():
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description='Delete bad/error-listed samples from dataset directories')
    parser.add_argument('--root', default=cfg['dataset']['root'],
                        help='Dataset root directory')
    parser.add_argument('--subdirs', nargs='+',
                        default=['HS/B', 'BF/A', 'BF/B'],
                        help='Subdirectories to clean (relative to --root)')
    parser.add_argument('--error-file', default=None,
                        help='Path to error list file (one ID per line)')
    parser.add_argument('--ids', nargs='+', default=None,
                        help='Sample IDs to delete (e.g. JX_18-4 JX_18-6)')
    parser.add_argument('--no-delay', action='store_true',
                        help='Skip the 3-second safety delay')
    args = parser.parse_args()

    if args.error_file is None and args.ids is None:
        parser.error("Must specify either --error-file or --ids")

    # Load IDs
    if args.error_file:
        if not os.path.exists(args.error_file):
            print(f"[Error] Error file not found: {args.error_file}")
            return
        with open(args.error_file, 'r', encoding='utf-8') as f:
            error_ids = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(error_ids)} IDs from {args.error_file}")
    else:
        error_ids = args.ids
        print(f"Received {len(error_ids)} IDs from command line")

    # Resolve full target directories
    target_dirs = [os.path.join(args.root, d) for d in args.subdirs]
    print(f"Target Root: {args.root}")
    print(f"Target Subdirs: {args.subdirs}")
    print("-" * 50)

    if not args.no_delay:
        print("Files will be PERMANENTLY deleted in 3 seconds...")
        time.sleep(3)

    deleted_count = 0
    not_found_count = 0

    for file_id in error_ids:
        filename = f"{file_id}.png"
        file_deleted = False

        for subdir in args.subdirs:
            dir_path = os.path.join(args.root, subdir)
            file_path = os.path.join(dir_path, filename)

            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"[Deleted] {subdir}/{filename}")
                    file_deleted = True
                    deleted_count += 1
                except Exception as e:
                    print(f"[Error] Failed to delete {file_path}: {e}")

        if not file_deleted:
            not_found_count += 1

    print("-" * 50)
    print("Cleanup Summary:")
    print(f"  Total files removed: {deleted_count}")
    print(f"  IDs not found:       {not_found_count}")

    # Post-deletion verification
    print("\nVerifying remaining file counts...")
    for subdir in args.subdirs:
        dir_path = os.path.join(args.root, subdir)
        if os.path.exists(dir_path):
            count = len([f for f in os.listdir(dir_path) if f.endswith('.png')])
            print(f"  {subdir}: {count} files remaining")


if __name__ == "__main__":
    main()
