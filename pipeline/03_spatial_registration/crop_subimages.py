"""
Interactive Manual Quality Filtering for BF Sub-Images
=======================================================
Paper reference: Section 3.1 Data Preparation

Generates 1440x1440 crops with 40% overlap from whole BF.png image,
displays them interactively for manual quality inspection.
User presses '1' to keep or '2' to skip each tile.

Input:  BF.png whole-slide image
Output: Selected tiles in BF_Subimages_Filtered/
"""
import os
import argparse
import cv2
import numpy as np

SUB_SIZE = 1440
OVERLAP = 0.4
OUTPUT_DIR_NAME = 'BF_Subimages_Filtered'


def generate_crops(h, w, sub_size, overlap):
    step = int(sub_size * (1 - overlap))
    y_starts = []
    y = 0
    while y < h:
        y_starts.append(y)
        if y + sub_size >= h:
            if y_starts[-1] != h - sub_size and h > sub_size:
                y_starts[-1] = h - sub_size
            break
        y += step

    x_starts = []
    x = 0
    while x < w:
        x_starts.append(x)
        if x + sub_size >= w:
            if x_starts[-1] != w - sub_size and w > sub_size:
                x_starts[-1] = w - sub_size
            break
        x += step

    return y_starts, x_starts


def process_filtering(gt_root, folders):
    print(f"Targeting specific folders: {folders}")

    valid_folders = []
    for name in folders:
        full_path = os.path.join(gt_root, name)
        if os.path.exists(full_path) and os.path.isdir(full_path):
            valid_folders.append(full_path)
        else:
            print(f"Warning: Folder '{name}' not found in {gt_root}")

    print(f"Found {len(valid_folders)} valid folders to process.")

    cv2.namedWindow("Filter Window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Filter Window", 1000, 1000)

    for folder_path in valid_folders:
        folder_name = os.path.basename(folder_path)
        bf_path = os.path.join(folder_path, 'BF.png')

        if not os.path.exists(bf_path):
            print(f"Skipping {folder_name}: BF.png not found.")
            continue

        save_dir = os.path.join(folder_path, OUTPUT_DIR_NAME)
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n--- Processing Folder: {folder_name} ---")
        large_img = cv2.imread(bf_path, cv2.IMREAD_GRAYSCALE)
        if large_img is None:
            print(f"Error: Could not read {bf_path}. Skipping.")
            continue

        h, w = large_img.shape
        y_starts, x_starts = generate_crops(h, w, SUB_SIZE, OVERLAP)
        total_crops = len(y_starts) * len(x_starts)
        count = 0

        print(f"Generated {total_crops} sub-grids.")
        print("Controls: '1' to KEEP | '2' to SKIP | 'ESC' to Quit")

        for y in y_starts:
            for x in x_starts:
                count += 1
                filename = f"sub_{y}_{x}.png"
                if os.path.exists(os.path.join(save_dir, filename)):
                    print(f"[{count}/{total_crops}] Already processed: {filename}")
                    continue

                sub_img = large_img[y:y + SUB_SIZE, x:x + SUB_SIZE]
                display_img = sub_img.copy()
                cv2.putText(display_img, f"{folder_name} | {count}/{total_crops}",
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
                cv2.putText(display_img, "Press 1: SAVE | Press 2: SKIP",
                            (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 255, 200), 3)

                cv2.imshow("Filter Window", display_img)

                while True:
                    key = cv2.waitKey(0)
                    if key == 27:
                        print("Exiting manually...")
                        cv2.destroyAllWindows()
                        return
                    elif key == ord('1'):
                        cv2.imwrite(os.path.join(save_dir, filename), sub_img)
                        print(f"Saved: {filename}")
                        break
                    elif key == ord('2'):
                        print(f"Skipped crop at y={y}, x={x}")
                        break

    cv2.destroyAllWindows()
    print("\nAll target folders processed.")


def main():
    parser = argparse.ArgumentParser(
        description='Interactive BF sub-image quality filtering')
    parser.add_argument('--gt_root', required=True,
                        help='Root directory containing GT-DATA folders with BF.png')
    parser.add_argument('--folders', nargs='+',
                        default=['1514619-A7', '1577035-C4', '1816663-A8'],
                        help='Folder names to process')
    args = parser.parse_args()
    process_filtering(args.gt_root, args.folders)


if __name__ == "__main__":
    main()
