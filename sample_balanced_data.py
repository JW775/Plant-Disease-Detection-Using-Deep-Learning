"""
File: sample_balanced_data.py
Responsible team member: ZhengWang
Description: Description: A Python script to perform random balanced sampling across dataset categories, ensuring a fixed number of images per class for training and validation splits.
"""
import os
import random
import shutil

# ==================== CONFIGURATION ====================
# 1. Source dataset directory (Path to your 3rd folder)
SOURCE_DIR = "D:/plant_data_resized_224"

# 2. Target dataset directory (Path to your 1st folder)
TARGET_DIR = "C:/Users/wangzheng/Desktop/Plant——Disease 素材/balance_data_2026.5.19"

# 3. Strictly limited sampling counts
TRAIN_TARGET = 1000
VAL_TARGET = 150
# =======================================================

def make_balanced_dataset():
    # 🎯 Set random seed for reproducibility (ensures the same images are selected each run)
    random.seed(42)

    if not os.path.exists(SOURCE_DIR):
        print(f"❌ Source directory not found: {SOURCE_DIR}. Please check the path!")
        return

    # Automatically detect if the validation folder is named 'valid' or 'val'
    src_val_name = "valid" if os.path.exists(os.path.join(SOURCE_DIR, "valid")) else "val"
    
    # Mapping: (source_split, target_count, target_split)
    tasks = [
        ("train", TRAIN_TARGET, "train"),
        (src_val_name, VAL_TARGET, "valid")  # Generates 'valid' directory in the target folder
    ]

    print("🚀 Smart balanced sampling system started...")

    for src_split, target_count, tgt_split in tasks:
        split_src_path = os.path.join(SOURCE_DIR, src_split)
        if not os.path.exists(split_src_path):
            print(f"⚠️ Split directory '{src_split}' not found. Skipping this task.")
            continue

        print(f"\n📂 Processing: [{src_split}] split -> Target set to {target_count} images per category")

        # Iterate through all plant disease categories
        for category in os.listdir(split_src_path):
            cat_src_path = os.path.join(split_src_path, category)
            if not os.path.isdir(cat_src_path):
                continue

            # Create corresponding output directories in the target folder
            cat_tgt_path = os.path.join(TARGET_DIR, tgt_split, category)
            if not os.path.exists(cat_tgt_path):
                os.makedirs(cat_tgt_path)

            # Get all valid images from the category folder
            all_imgs = [f for f in os.listdir(cat_src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # Execute random sampling without replacement
            if len(all_imgs) < target_count:
                print(f"⚠️ Warning: [{category}] has only {len(all_imgs)} images, which is less than {target_count}. Copying all available images.")
                sampled_imgs = all_imgs
            else:
                sampled_imgs = random.sample(all_imgs, target_count)

            # Copy files safely
            copied_count = 0
            for img_name in sampled_imgs:
                src_file = os.path.join(cat_src_path, img_name)
                tgt_file = os.path.join(cat_tgt_path, img_name)
                try:
                    shutil.copy2(src_file, tgt_file)
                    copied_count += 1
                except Exception as e:
                    print(f"⚠️ Failed to copy image {img_name}: {e}")
            
            print(f"  └─ Category [{category}]: Successfully sampled and saved {copied_count} images")

    print(f"\n🎉 Done! The balanced dataset has been generated successfully at:\n👉 {TARGET_DIR}")

if __name__ == '__main__':
    make_balanced_dataset()
import os
import random
import shutil

# ==================== CONFIGURATION ====================
# 1. Source dataset directory (Path to your 3rd folder)
SOURCE_DIR = "D:/plant_data_resized_224"

# 2. Target dataset directory (Path to your 1st folder)
TARGET_DIR = "C:/Users/wangzheng/Desktop/Plant——Disease 素材/balance_data_2026.5.19"

# 3. Strictly limited sampling counts
TRAIN_TARGET = 1000
VAL_TARGET = 150
# =======================================================

def make_balanced_dataset():
    # 🎯 Set random seed for reproducibility (ensures the same images are selected each run)
    random.seed(42)

    if not os.path.exists(SOURCE_DIR):
        print(f"❌ Source directory not found: {SOURCE_DIR}. Please check the path!")
        return

    # Automatically detect if the validation folder is named 'valid' or 'val'
    src_val_name = "valid" if os.path.exists(os.path.join(SOURCE_DIR, "valid")) else "val"
    
    # Mapping: (source_split, target_count, target_split)
    tasks = [
        ("train", TRAIN_TARGET, "train"),
        (src_val_name, VAL_TARGET, "valid")  # Generates 'valid' directory in the target folder
    ]

    print("🚀 Smart balanced sampling system started...")

    for src_split, target_count, tgt_split in tasks:
        split_src_path = os.path.join(SOURCE_DIR, src_split)
        if not os.path.exists(split_src_path):
            print(f"⚠️ Split directory '{src_split}' not found. Skipping this task.")
            continue

        print(f"\n📂 Processing: [{src_split}] split -> Target set to {target_count} images per category")

        # Iterate through all plant disease categories
        for category in os.listdir(split_src_path):
            cat_src_path = os.path.join(split_src_path, category)
            if not os.path.isdir(cat_src_path):
                continue

            # Create corresponding output directories in the target folder
            cat_tgt_path = os.path.join(TARGET_DIR, tgt_split, category)
            if not os.path.exists(cat_tgt_path):
                os.makedirs(cat_tgt_path)

            # Get all valid images from the category folder
            all_imgs = [f for f in os.listdir(cat_src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # Execute random sampling without replacement
            if len(all_imgs) < target_count:
                print(f"⚠️ Warning: [{category}] has only {len(all_imgs)} images, which is less than {target_count}. Copying all available images.")
                sampled_imgs = all_imgs
            else:
                sampled_imgs = random.sample(all_imgs, target_count)

            # Copy files safely
            copied_count = 0
            for img_name in sampled_imgs:
                src_file = os.path.join(cat_src_path, img_name)
                tgt_file = os.path.join(cat_tgt_path, img_name)
                try:
                    shutil.copy2(src_file, tgt_file)
                    copied_count += 1
                except Exception as e:
                    print(f"⚠️ Failed to copy image {img_name}: {e}")
            
            print(f"  └─ Category [{category}]: Successfully sampled and saved {copied_count} images")

    print(f"\n🎉 Done! The balanced dataset has been generated successfully at:\n👉 {TARGET_DIR}")

if __name__ == '__main__':
    make_balanced_dataset()
import os
import random
import shutil

# ==================== CONFIGURATION ====================
# 1. Source dataset directory (Path to your 3rd folder)
SOURCE_DIR = "D:/plant_data_resized_224"

# 2. Target dataset directory (Path to your 1st folder)
TARGET_DIR = "C:/Users/wangzheng/Desktop/Plant——Disease 素材/balance_data_2026.5.19"

# 3. Strictly limited sampling counts
TRAIN_TARGET = 1000
VAL_TARGET = 150
# =======================================================

def make_balanced_dataset():
    # 🎯 Set random seed for reproducibility (ensures the same images are selected each run)
    random.seed(42)

    if not os.path.exists(SOURCE_DIR):
        print(f"❌ Source directory not found: {SOURCE_DIR}. Please check the path!")
        return

    # Automatically detect if the validation folder is named 'valid' or 'val'
    src_val_name = "valid" if os.path.exists(os.path.join(SOURCE_DIR, "valid")) else "val"
    
    # Mapping: (source_split, target_count, target_split)
    tasks = [
        ("train", TRAIN_TARGET, "train"),
        (src_val_name, VAL_TARGET, "valid")  # Generates 'valid' directory in the target folder
    ]

    print("🚀 Smart balanced sampling system started...")

    for src_split, target_count, tgt_split in tasks:
        split_src_path = os.path.join(SOURCE_DIR, src_split)
        if not os.path.exists(split_src_path):
            print(f"⚠️ Split directory '{src_split}' not found. Skipping this task.")
            continue

        print(f"\n📂 Processing: [{src_split}] split -> Target set to {target_count} images per category")

        # Iterate through all plant disease categories
        for category in os.listdir(split_src_path):
            cat_src_path = os.path.join(split_src_path, category)
            if not os.path.isdir(cat_src_path):
                continue

            # Create corresponding output directories in the target folder
            cat_tgt_path = os.path.join(TARGET_DIR, tgt_split, category)
            if not os.path.exists(cat_tgt_path):
                os.makedirs(cat_tgt_path)

            # Get all valid images from the category folder
            all_imgs = [f for f in os.listdir(cat_src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # Execute random sampling without replacement
            if len(all_imgs) < target_count:
                print(f"⚠️ Warning: [{category}] has only {len(all_imgs)} images, which is less than {target_count}. Copying all available images.")
                sampled_imgs = all_imgs
            else:
                sampled_imgs = random.sample(all_imgs, target_count)

            # Copy files safely
            copied_count = 0
            for img_name in sampled_imgs:
                src_file = os.path.join(cat_src_path, img_name)
                tgt_file = os.path.join(cat_tgt_path, img_name)
                try:
                    shutil.copy2(src_file, tgt_file)
                    copied_count += 1
                except Exception as e:
                    print(f"⚠️ Failed to copy image {img_name}: {e}")
            
            print(f"  └─ Category [{category}]: Successfully sampled and saved {copied_count} images")

    print(f"\n🎉 Done! The balanced dataset has been generated successfully at:\n👉 {TARGET_DIR}")

if __name__ == '__main__':
    make_balanced_dataset()
import os
import random
import shutil

# ==================== CONFIGURATION ====================
# 1. Source dataset directory (Path to your 3rd folder)
SOURCE_DIR = "D:/plant_data_resized_224"

# 2. Target dataset directory (Path to your 1st folder)
TARGET_DIR = "C:/Users/wangzheng/Desktop/Plant——Disease 素材/balance_data_2026.5.19"

# 3. Strictly limited sampling counts
TRAIN_TARGET = 1000
VAL_TARGET = 150
# =======================================================

def make_balanced_dataset():
    # 🎯 Set random seed for reproducibility (ensures the same images are selected each run)
    random.seed(42)

    if not os.path.exists(SOURCE_DIR):
        print(f"❌ Source directory not found: {SOURCE_DIR}. Please check the path!")
        return

    # Automatically detect if the validation folder is named 'valid' or 'val'
    src_val_name = "valid" if os.path.exists(os.path.join(SOURCE_DIR, "valid")) else "val"
    
    # Mapping: (source_split, target_count, target_split)
    tasks = [
        ("train", TRAIN_TARGET, "train"),
        (src_val_name, VAL_TARGET, "valid")  # Generates 'valid' directory in the target folder
    ]

    print("🚀 Smart balanced sampling system started...")

    for src_split, target_count, tgt_split in tasks:
        split_src_path = os.path.join(SOURCE_DIR, src_split)
        if not os.path.exists(split_src_path):
            print(f"⚠️ Split directory '{src_split}' not found. Skipping this task.")
            continue

        print(f"\n📂 Processing: [{src_split}] split -> Target set to {target_count} images per category")

        # Iterate through all plant disease categories
        for category in os.listdir(split_src_path):
            cat_src_path = os.path.join(split_src_path, category)
            if not os.path.isdir(cat_src_path):
                continue

            # Create corresponding output directories in the target folder
            cat_tgt_path = os.path.join(TARGET_DIR, tgt_split, category)
            if not os.path.exists(cat_tgt_path):
                os.makedirs(cat_tgt_path)

            # Get all valid images from the category folder
            all_imgs = [f for f in os.listdir(cat_src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # Execute random sampling without replacement
            if len(all_imgs) < target_count:
                print(f"⚠️ Warning: [{category}] has only {len(all_imgs)} images, which is less than {target_count}. Copying all available images.")
                sampled_imgs = all_imgs
            else:
                sampled_imgs = random.sample(all_imgs, target_count)

            # Copy files safely
            copied_count = 0
            for img_name in sampled_imgs:
                src_file = os.path.join(cat_src_path, img_name)
                tgt_file = os.path.join(cat_tgt_path, img_name)
                try:
                    shutil.copy2(src_file, tgt_file)
                    copied_count += 1
                except Exception as e:
                    print(f"⚠️ Failed to copy image {img_name}: {e}")
            
            print(f"  └─ Category [{category}]: Successfully sampled and saved {copied_count} images")

    print(f"\n🎉 Done! The balanced dataset has been generated successfully at:\n👉 {TARGET_DIR}")

if __name__ == '__main__':
    make_balanced_dataset()
