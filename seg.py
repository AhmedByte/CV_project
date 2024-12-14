import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_lbp(image):
    """
    Computes the Local Binary Pattern (LBP) manually for a given grayscale image.
    """
    lbp_image = np.zeros_like(image, dtype=np.uint8)
    rows, cols = image.shape
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            center_pixel = image[i, j]
            pixel7 = int(image[i-1, j-1] > center_pixel)
            pixel6 = int(image[i-1, j] > center_pixel)
            pixel5 = int(image[i-1, j+1] > center_pixel)
            pixel4 = int(image[i, j+1] > center_pixel)
            pixel3 = int(image[i+1, j+1] > center_pixel)
            pixel2 = int(image[i+1, j] > center_pixel)
            pixel1 = int(image[i+1, j-1] > center_pixel)
            pixel0 = int(image[i, j-1] > center_pixel)
            lbp_value = (pixel7 << 7) | (pixel6 << 6) | (pixel5 << 5) | (pixel4 << 4) | \
                        (pixel3 << 3) | (pixel2 << 2) | (pixel1 << 1) | pixel0
            lbp_image[i, j] = lbp_value
    
    return lbp_image

def segment_image_grabcut(image):
    mask = np.zeros(image.shape[:2], np.uint8)
    h, w = image.shape[:2]
    rect = (10, 10, w - 20, h - 20)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = image * mask2[:, :, np.newaxis]
    return result

def segment_image_watershed(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    segmented = image.copy()
    segmented[markers == -1] = [255, 0, 0]
    return segmented
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


def display_images(original, grabcut, watershed, region, lbp):
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 5, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")
    plt.subplot(1, 5, 2)
    plt.imshow(cv2.cvtColor(grabcut, cv2.COLOR_BGR2RGB))
    plt.title("GrabCut")
    plt.axis("off")
    plt.subplot(1, 5, 3)
    plt.imshow(cv2.cvtColor(watershed, cv2.COLOR_BGR2RGB))
    plt.title("Watershed")
    plt.axis("off")
    plt.subplot(1, 5, 4)
    plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
    plt.title("Region-Based")
    plt.axis("off")
    plt.subplot(1, 5, 5)
    plt.imshow(lbp, cmap="gray")
    plt.title("LBP")
    plt.axis("off")
    plt.show()

# Load the image
image_path = "train_data/apple/Apple (33).jpeg"
original_image = cv2.imread(image_path)
grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Perform segmentation
segmented_grabcut = segment_image_grabcut(original_image)
segmented_watershed = segment_image_watershed(original_image)
seed_point = (50, 50)  # You can adjust the seed point for Region Growing
segmented_region = region_based_segmentation(original_image)
lbp_image = compute_lbp(grayscale_image)

# Display results
display_images(original_image, segmented_grabcut, segmented_watershed, segmented_region, lbp_image)
