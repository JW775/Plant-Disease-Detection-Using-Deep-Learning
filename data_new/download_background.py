"""
File: download_background.py
Responsible team member: ZhengWang
Description: Description: A Python script to automatically download 224x224 non-plant background images from the Picsum API to build a negative control class for training and validation splits.
"""
import os
import requests
import time

# 1. Make sure the script runs in the current directory (same level as train and valid)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else "."

# Match the folder names and triple underscore format
TRAIN_BG_DIR = os.path.join(CURRENT_DIR, "train", "Background___non_plant")
VALID_BG_DIR = os.path.join(CURRENT_DIR, "valid", "Background___non_plant")

# Automatically create these folders
os.makedirs(TRAIN_BG_DIR, exist_ok=True)
os.makedirs(VALID_BG_DIR, exist_ok=True)

def download_images(target_dir, num_images, start_seed):
    print(f"Downloading {num_images} diverse background images (size: 224x224) to: {target_dir}...")
    success_count = 0
    current_seed = start_seed
    
    while success_count < num_images:
        # 224x224 strictly matches your existing crop image size
        url = f"https://picsum.photos/seed/{current_seed}/224/224"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                file_name = f"bg_{current_seed}.jpg"
                with open(os.path.join(target_dir, file_name), "wb") as f:
                    f.write(response.content)
                success_count += 1
                if success_count % 50 == 0:
                    print(f"Completed: {success_count}/{num_images}")
        except Exception:
            pass # Skip network errors and try the next seed
        
        current_seed += 1
        time.sleep(0.01) # Smaller size downloads faster, shorten delay time

print("================ Start building class 45 ================")
# Training set: 1000 images
download_images(TRAIN_BG_DIR, 1000, start_seed=100)

# Validation set: 150 images (different seed range to avoid duplicates)
download_images(VALID_BG_DIR, 150, start_seed=5000)

print("================ Class 45 (224x224) built successfully! ================")
