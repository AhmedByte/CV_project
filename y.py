import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# Paths
data_path = "Ptrain_data"
classes = ["apple", "banana", "mango", "grape", "strawberry"]

# Segmentation Function

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

# LBP Feature Extraction
def extract_lbp_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp_image = np.zeros_like(gray)
    rows, cols = gray.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            center = gray[i, j]
            binary_string = ''.join(['1' if gray[i + di, j + dj] > center else '0'
                                     for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]] )
            lbp_image[i, j] = int(binary_string, 2)
    return lbp_image.flatten()

# Dataset Class
class FruitDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform

        for label, class_name in enumerate(classes):
            class_path = os.path.join(data_path, class_name)
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                image = cv2.imread(image_path)
                if image is not None:
                    segmented = segment_image_grabcut(image)
                    features = extract_lbp_features(segmented)
                    self.data.append(features)
                    self.labels.append(label)

        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

# Neural Network Class
class EnhancedNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EnhancedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

# Data Augmentation and Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

# Data Preparation
dataset = FruitDataset(data_path, transform=transform)
data_train, data_test, labels_train, labels_test = train_test_split(
    dataset.data, dataset.labels, test_size=0.2, random_state=42)

train_loader = DataLoader(list(zip(data_train, labels_train)), batch_size=32, shuffle=True)
test_loader = DataLoader(list(zip(data_test, labels_test)), batch_size=32, shuffle=False)

# Model Training
input_size = dataset.data.shape[1]
num_classes = len(classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EnhancedNN(input_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # During the training loop, when displaying the image
        if idx % 50 == 0:
            img = features[0].cpu().numpy()

            # Reshape the image to its original shape (128x128 in this case)
            img = img.reshape(128, 128)  # Adjust based on your image size

            plt.imshow(img, cmap='gray')
            plt.title(f"Epoch [{epoch + 1}/{epochs}], Batch {idx + 1}")
            plt.show()


    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

# Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))
# After training loop, save the model
torch.save(model.state_dict(), "fruit_model.pth")

