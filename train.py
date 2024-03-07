from functools import total_ordering
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from google.cloud import storage

# Create a client to interact with the Google Cloud Storage service
client = storage.Client()

# Specify the names of your buckets
bucket_names = ["healthy_lungs", "viral_pneumonia", "bacterial_pneumonia", "lung_coivd", "bronchitis_lungs"]

# Download the dataset files from Google Cloud Storage
for bucket_name in bucket_names:
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs()

    # Create a directory for each bucket (class)
    os.makedirs(bucket_name, exist_ok=True)

    for blob in blobs:
        destination_file = os.path.join(bucket_name, blob.name)
        if not os.path.exists(destination_file):
            blob.download_to_filename(destination_file)
            print(f"Downloaded {destination_file}")
        else:
            print(f"File {destination_file} already exists. Skipping download.")

# Add code here to load and preprocess your dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def is_valid_file(path):
    supported_extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp']
    return os.path.splitext(path)[-1].lower() in supported_extensions

class RemappedImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, is_valid_file=None, class_directories=None):
        self.class_directories = class_directories
        super().__init__(root, transform=transform, target_transform=target_transform, is_valid_file=is_valid_file)

    def find_classes(self, dir):
        if self.class_directories is not None:
            classes = self.class_directories
        else:
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        remapped_target = self.class_to_idx[self.classes[target]]
        return sample, remapped_target

dataset = RemappedImageFolder(
    root=".",
    transform=transform,
    is_valid_file=is_valid_file,
    class_directories=["healthy_lungs", "viral_pneumonia", "bacterial_pneumonia", "lung_coivd", "bronchitis_lungs"]
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print("Class names and labels:")
print(dataset.class_to_idx)

# Define the neural network architecture
class LungNet(nn.Module):
    def __init__(self):
        super(LungNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, len(dataset.classes))  # Output units based on the number of classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    # Create an instance of the neural network
    model = LungNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    # Save the trained model
    torch.save(model.state_dict(), "lung_model.pth")
