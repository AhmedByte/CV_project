import os
import cv2
import shutil
import numpy as np

# Paths to datasets
base_dirs = {
    "train": "train",
    "test": "test",
    "valid": "valid"
}

# Path to store the merged images (after preprocessing)
merged_data_dir = "data"

# Categories
categories = ["apple", "banana", "grape", "mango", "strawberry"]

# Fixed size for resizing
IMG_SIZE = (128, 128)

# Step 1: Merge all images into one folder with labels as subfolders
def merge_images_into_one_folder():
    if not os.path.exists(merged_data_dir):
        os.makedirs(merged_data_dir)

    for category in categories:
        category_dir = os.path.join(merged_data_dir, category)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
        
        # Merge images from each dataset (train, test, valid)
        for set_type in base_dirs:
            set_dir = base_dirs[set_type]
            category_folder = os.path.join(set_dir, category)
            
            if os.path.exists(category_folder):
                for img_name in os.listdir(category_folder):
                    img_path = os.path.join(category_folder, img_name)
                    if os.path.isfile(img_path):
                        # Copy the image to the merged folder
                        shutil.copy(img_path, os.path.join(category_dir, img_name))
                        
merge_images_into_one_folder()