import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Path to your merged data
merged_data_dir = "data"

# Path to save the processed images
processed_data_dir = "new_data"
if not os.path.exists(processed_data_dir):
    os.makedirs(processed_data_dir)

# Paths for the new directories for train/test data
train_data_dir = "train_data"
test_data_dir = "test_data"
if not os.path.exists(train_data_dir):
    os.makedirs(train_data_dir)
if not os.path.exists(test_data_dir):
    os.makedirs(test_data_dir)

# Categories of the images
categories = ["apple", "banana", "grape", "mango", "strawberry"]

# Fixed size for resizing
IMG_SIZE = (128, 128)

# Step 1: Preprocess images (resize, noise reduction, normalization, contrast enhancement)
def preprocess_images():
    # Loop through each category
    for category in categories:
        # Paths for category folders
        category_dir = os.path.join(merged_data_dir, category)
        category_processed_dir = os.path.join(processed_data_dir, category)
        
        # Create processed folder for each category
        if not os.path.exists(category_processed_dir):
            os.makedirs(category_processed_dir)

        # Get all the image filenames in the current category folder
        images = os.listdir(category_dir)

        # Process each image
        for img_name in images:
            img_path = os.path.join(category_dir, img_name)
            img = cv2.imread(img_path)

            if img is not None:
                # Resize image
                resized_img = cv2.resize(img, IMG_SIZE)
                
                # Noise reduction using Gaussian blur (apply on the original color image)
                blurred_img = cv2.GaussianBlur(resized_img, (5, 5), 0)
                
                # Normalization: Scale pixel values to range [0, 1]
                normalized_img = blurred_img / 255.0

                # Split the image into its R, G, B channels
                (B, G, R) = cv2.split(normalized_img)

                # Apply histogram equalization to each channel
                R_eq = cv2.equalizeHist((R * 255).astype(np.uint8))  # Convert to uint8 for equalizeHist
                G_eq = cv2.equalizeHist((G * 255).astype(np.uint8))  # Convert to uint8 for equalizeHist
                B_eq = cv2.equalizeHist((B * 255).astype(np.uint8))  # Convert to uint8 for equalizeHist

                # Merge the channels back
                enhanced_img = cv2.merge([B_eq, G_eq, R_eq])

                # Save the processed image in the new folder
                processed_img_path = os.path.join(category_processed_dir, img_name)
                
                # Check if the image was successfully saved
                if cv2.imwrite(processed_img_path, enhanced_img):
                    print(f"Image {img_name} saved successfully!")
                else:
                    print(f"Error saving image {img_name}")

    print("Preprocessing complete and images saved in new folders!")

# Step 2: Split data into train and test sets (80% train, 20% test)
def split_data():
    # Loop through each category
    for category in categories:
        # Paths for category folders
        category_processed_dir = os.path.join(processed_data_dir, category)
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
preprocess_images()  # Process the images first
split_data()         # Split the processed images into train/test sets
