import os
from PIL import Image

# ================= Configuration =================
# Path to your original 5GB dataset
SOURCE_DIR = "D:/plant_data_final"  
# Path for the resized images (program will create this folder)
TARGET_DIR = "D:/plant_data_resized_224"  
# Target image size
TARGET_SIZE = (224, 224)
# =================================================

def process_images():
    # Counter for processed images
    count = 0
    
    # Traverse through source directory and subdirectories
    for root, dirs, files in os.walk(SOURCE_DIR):
        # Map source directory structure to target directory
        relative_path = os.path.relpath(root, SOURCE_DIR)
        target_path = os.path.join(TARGET_DIR, relative_path)
        
        # Create subfolders if they don't exist
        if not os.path.exists(target_path):
            os.makedirs(target_path)
            
        for file in files:
            # Only process image files
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_file = os.path.join(root, file)
                dst_file = os.path.join(target_path, file)
                
                try:
                    # 1. Open the original image
                    with Image.open(src_file) as img:
                        # 2. Convert to RGB mode to avoid errors
                        img = img.convert('RGB')
                        # 3. Resize the image
                        img_resized = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                        # 4. Save to target folder with 85% quality to save space
                        img_resized.save(dst_file, format='JPEG', quality=85)
                    
                    count += 1
                    # Print progress every 5000 images
                    if count % 5000 == 0:
                        print(f"Successfully processed {count} images...")
                        
                except Exception as e:
                    print(f"Failed to process: {src_file}, Error: {e}")

    print(f"\n🎉 Task completed! Total images processed: {count}")
    print(f"Please check the standard data in: {TARGET_DIR}")

if __name__ == '__main__':
    print("Starting batch processing pipeline, please wait...")
    process_images()