import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from y import FruitDataset, EnhancedNN, classes  # Adjust import as needed

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a new instance of the model
input_size = 16384  # This should match the input size used during training (LBP features)
num_classes = len(classes)
model = EnhancedNN(input_size, num_classes).to(device)

# Load the trained model weights (this will **not retrain the model**, just loads it)
model.load_state_dict(torch.load("fruit_model.pth"))
model.eval()  # Set model to evaluation mode (important)

# Prepare the test data
test_data_path = "test_data"
dataset = FruitDataset(test_data_path)
data_test, labels_test = dataset.data, dataset.labels

# Create DataLoader for the test set
test_loader = DataLoader(list(zip(data_test, labels_test)), batch_size=32, shuffle=False)

# Display images and predictions for each class
fig, axes = plt.subplots(5, 3, figsize=(10, 15))  # 5 rows (for each class), 3 images per row

# Iterate through each class
for class_idx, class_name in enumerate(classes):
    # Get 3 samples of each class
    images_for_class = []
    labels_for_class = []
    predictions_for_class = []

    for i, (image, label) in enumerate(zip(data_test, labels_test)):
        if label == class_idx and len(images_for_class) < 3:
            images_for_class.append(image)
            labels_for_class.append(label)

            # Get the prediction for this image
            image_tensor = torch.tensor(image).float().unsqueeze(0).to(device)
            output = model(image_tensor)
            _, predicted_label = torch.max(output, 1)
            predictions_for_class.append(predicted_label.item())

    # Plot the images with predictions
    for i, (img, pred) in enumerate(zip(images_for_class, predictions_for_class)):
        ax = axes[class_idx, i]
        img = img.reshape(28, 28)  # Adjust if needed to your image size (e.g., 28x28)
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Pred: {classes[pred]}")
        ax.axis('off')

plt.tight_layout()
plt.show()

# Evaluate the model on the test set and print classification report
all_preds = []
all_labels = []

with torch.no_grad():  # No gradient calculations needed for evaluation
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)

        # Forward pass
        outputs = model(features)
        _, preds = torch.max(outputs, 1)

        # Store predictions and labels
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate accuracy and classification report
accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"Test Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))
