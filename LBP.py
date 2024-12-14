
import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

image_path = 'train_data/apple/Apple (1).png'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image, (256, 256))
pixels = image_resized.reshape(-1, 3)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(pixels)
segmented_image = kmeans.labels_.reshape(image_resized.shape[:2])
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image_resized)
plt.subplot(1, 2, 2)
plt.title('Segmented Image')
plt.imshow(segmented_image, cmap='viridis')
plt.show()
segmented_gray = (segmented_image * (255 // (kmeans.n_clusters - 1))).astype(np.uint8)

def extract_texture_features(image_gray):
    glcm = greycomatrix(image_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    features = {
        'contrast': greycoprops(glcm, 'contrast').mean(),
        'dissimilarity': greycoprops(glcm, 'dissimilarity').mean(),
        'homogeneity': greycoprops(glcm, 'homogeneity').mean(),
        'energy': greycoprops(glcm, 'energy').mean(),
        'correlation': greycoprops(glcm, 'correlation').mean(),
    }
    return features

features = extract_texture_features(segmented_gray)
print("Extracted Texture Features:")
for key, value in features.items():
    print(f"{key}: {value:.4f}")
