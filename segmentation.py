import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function for Region-Based Segmentation (Watershed)
def region_based_segmentation(image):
    """
    Performs region-based segmentation using the Watershed algorithm.
    
    Args:
        image (numpy.ndarray): Input image (color).
    
    Returns:
        numpy.ndarray: Segmented image with boundaries highlighted.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Binary thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Define background and foreground areas
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    # Find unknown regions
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Apply Watershed
    markers = cv2.watershed(image, markers)
    segmented_image = image.copy()
    segmented_image[markers == -1] = [255, 0, 0]  # Mark boundaries in red
    
    return segmented_image

# Function for LBP Feature Extraction
def compute_lbp(image):
    """
    Computes the Local Binary Pattern (LBP) manually for a given grayscale image.
    
    Args:
        image (numpy.ndarray): Grayscale image.
    
    Returns:
        numpy.ndarray: Image with LBP applied.
    """
    # Create an empty framework matrix for LBP
    lbp_image = np.zeros_like(image, dtype=np.uint8)
    
    # Get dimensions of the image
    rows, cols = image.shape
    
    # Loop over each pixel in the image (excluding the borders)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Center pixel
            center_pixel = image[i, j]
            
            # Compute the binary pattern by comparing neighbors with the center pixel
            pixel7 = int(image[i-1, j-1] > center_pixel)            
            pixel6 = int(image[i-1, j] > center_pixel)    
            pixel5 = int(image[i-1, j+1] > center_pixel)               
            pixel4 = int(image[i, j+1] > center_pixel)            
            pixel3 = int(image[i+1, j+1] > center_pixel)         
            pixel2 = int(image[i+1, j] > center_pixel) 
            pixel1 = int(image[i+1, j-1] > center_pixel)
            pixel0 = int(image[i, j-1] > center_pixel)  
            
            # Compute the LBP value
            lbp_value = (pixel7 << 7) | (pixel6 << 6) | (pixel5 << 5) | (pixel4 << 4) | \
                        (pixel3 << 3) | (pixel2 << 2) | (pixel1 << 1) | pixel0
            
            # Store the LBP value in the new image
            lbp_image[i, j] = lbp_value
    
    return lbp_image

# Main Code
if __name__ == "__main__":
    # Step 1: Load the image
    image_path = "data/mango/Mango (111).jpeg"  # Replace with your image path
    img = cv2.imread(image_path)
    
    # Step 2: Perform Segmentation
    segmented_image = region_based_segmentation(img)
    
    # Step 3: Perform LBP Feature Extraction
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp_image = compute_lbp(gray)
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    
    # Segmented Image
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
    plt.title("Segmented Image")
    plt.axis("off")
    
    # LBP Image
    plt.subplot(1, 3, 3)
    plt.imshow(lbp_image, cmap="gray")
    plt.title("LBP Features")
    plt.axis("off")
    
    plt.show()
