import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# Main code
if __name__ == "__main__":
    # Step 1: Load the image
    image_path = "test_data/apple/Apple (2).png"  # Replace with your image path
    img = cv2.imread(image_path)
    
    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 3: Compute LBP
    lbp_image = compute_lbp(gray)
    
    # Visualization
    plt.figure(figsize=(10, 5))
    
    # Original grayscale image
    plt.subplot(1, 2, 1)
    plt.imshow(gray, cmap="gray")
    plt.title("Original Grayscale Image")
    plt.axis("off")
    
    # LBP Image
    plt.subplot(1, 2, 2)
    plt.imshow(lbp_image, cmap="gray")
    plt.title("LBP Image")
    plt.axis("off")
    
    plt.show()
