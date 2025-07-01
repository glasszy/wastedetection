import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import os

class WasteClassifierCNN(nn.Module):
    def __init__(self, num_classes=22):  # Number of waste categories
        super(WasteClassifierCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # First convolution
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Second convolution
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.25)

        # Adjust the input size of fc1 based on the convolutional layer output
        # After two max-pooling layers, each of size (64, 64) -> (32, 32) -> (16, 16)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Adjusted for correct flattened size
        self.fc2 = nn.Linear(128, num_classes)  # Output layer

    def forward(self, x):
        # Convolutional layers with ReLU activations and pooling
        x = self.pool(F.relu(self.conv1(x)))  # 64x64 -> 32x32
        x = self.pool(F.relu(self.conv2(x)))  # 32x32 -> 16x16
        
        # Flatten the tensor
        x = x.view(-1, 64 * 16 * 16)  # Adjusted for correct flattened size

        # Fully connected layers with dropout
        x = self.dropout(F.relu(self.fc1(x)))  # Fully connected layer
        x = self.fc2(x)  # Output layer
        
        return x

# Load the trained model
model = WasteClassifierCNN(num_classes=22)
model.load_state_dict(torch.load("waste_cnn_model.pth"))
model.eval()  # Set the model to evaluation mode

# Define the transformation (ensure it matches training)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Load the test dataset (you may want to specify your test folder)
test_data = datasets.ImageFolder('cnn_dataset', transform=transform)  # Replace with actual test path
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Evaluate the model
correct = 0
total = 0
with torch.no_grad():  # Disable gradient calculation for evaluation
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy on test dataset: {accuracy:.2f}%")
