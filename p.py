import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Path to your merged data
merged_data_dir = "data"


# Paths for the new directories for train/test data
train_data_dir = "Ptrain_data"
test_data_dir = "Ptest_data"
if not os.path.exists(train_data_dir):
    os.makedirs(train_data_dir)
if not os.path.exists(test_data_dir):
    os.makedirs(test_data_dir)

# Categories of the images
categories = ["apple", "banana", "grape", "mango", "strawberry"]


# Step 2: Split data into train and test sets (80% train, 20% test)
def split_data():
    # Loop through each category
    for category in categories:
        # Paths for category folders
        category_processed_dir = os.path.join(merged_data_dir, category)
        category_train_dir = os.path.join(train_data_dir, category)
        category_test_dir = os.path.join(test_data_dir, category)

        if not os.path.exists(category_train_dir):
            os.makedirs(category_train_dir)
        if not os.path.exists(category_test_dir):
            os.makedirs(category_test_dir)

        # Get all the processed image filenames in the current category folder
        images = os.listdir(category_processed_dir)

        # Split the images into training and testing sets (80% train, 20% test)
        train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

        # Move images to the corresponding train and test directories
        for img_name in train_images:
            shutil.move(os.path.join(category_processed_dir, img_name), os.path.join(category_train_dir, img_name))

        for img_name in test_images:
            shutil.move(os.path.join(category_processed_dir, img_name), os.path.join(category_test_dir, img_name))

    print("Data split into train and test sets successfully!")

# Execute preprocessing and splitting
split_data()         # Split the processed images into train/test sets
