"""
Train K-Means on Target Domain for Mask Loss
=============================================
Paper reference: Section 3.2 Loss Functions (soft-assignment)

Trains a K-Means clustering model on target domain H&E images (trainB)
to obtain color cluster centroids. These centroids are used in
cut_model.py for the differentiable soft-assignment mask loss:

  P_bg = softmax(-d_bg / T)  where d_bg = ||pixel - centroid_bg||^2

The brightest cluster (highest RGB sum) is automatically identified
as the background class. Use --bg_label <id> in training if auto-
detection is incorrect.

Input:  Target domain images (trainB/)
Output: kmeans_model.pkl (scikit-learn KMeans model)
"""
import os
import numpy as np
import joblib
from sklearn.cluster import KMeans
import argparse
from PIL import Image

# Set up argument parsing
parser = argparse.ArgumentParser(description='Train K-Means on Linux using PIL')
parser.add_argument('--input_dir', type=str, default='./datasets/Target_Data/trainB', 
                    help='path to trainB images (HE style)')
parser.add_argument('--output_path', type=str, default='./k_means/kmeans_model.pkl', 
                    help='path to save the model')
parser.add_argument('--n_clusters', type=int, default=5, help='number of clusters')
args = parser.parse_args()

def train_kmeans():
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    all_pixels = []
    # Support common image extensions
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist! Please check --input_dir.")
        return

    # List and sort files
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(valid_extensions)]
    image_files.sort()
    
    # Optional: Limit the number of training images to save memory
    # image_files = image_files[:100] 

    print(f"Reading {len(image_files)} images for K-Means training (using PIL)...")
    
    if len(image_files) == 0:
        print("Error: No images found in the directory!")
        return

    for i, filename in enumerate(image_files):
        img_path = os.path.join(args.input_dir, filename)
        
        try:
            # 1. Open image using PIL
            img = Image.open(img_path).convert('RGB')
            
            # 2. Downsampling: Resize image to speed up K-Means training
            # K-Means only needs color distribution; 256x256 is sufficient.
            img = img.resize((256, 256), Image.NEAREST)
            
            # 3. Convert to numpy array
            img_np = np.array(img)
            
            # 4. Flatten pixels to (H*W, 3)
            pixels = img_np.reshape((-1, 3))
            all_pixels.append(pixels)
            
            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{len(image_files)} images")
                
        except Exception as e:
            print(f"Skipping corrupted or unreadable file {filename}: {e}")
            continue

    if not all_pixels:
        print("Error: No valid pixel data collected.")
        return

    # Concatenate data
    print("Concatenating data...")
    X_train = np.vstack(all_pixels)
    
    # Train K-Means
    print(f"Training K-means (n_clusters={args.n_clusters})... Data shape: {X_train.shape}")
    # n_init=10 is the default in newer sklearn versions, explicit setting is safer
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_train)

    # Save model
    joblib.dump(kmeans, args.output_path)
    print(f"✅ Model successfully saved to: {args.output_path}")
    
    # Verify Background Class
    print("-" * 30)
    print("Cluster Centers (RGB, 0-255):")
    centers = kmeans.cluster_centers_
    print(centers)
    print("-" * 30)
    
    # Automatically find the brightest cluster center (usually background)
    # Calculate brightness by summing RGB channels
    brightness = np.sum(centers, axis=1)
    bg_label = np.argmax(brightness)
    
    print(f"Inferred Background Label (Bg Label): {bg_label}")
    print(f"Corresponding RGB Center: {centers[bg_label]}")
    print("Please check if this center is close to white [255, 255, 255].")
    print(f"If yes, use this in your training command: --bg_label {bg_label}")

if __name__ == '__main__':
    train_kmeans()