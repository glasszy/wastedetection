import os
import cv2
from ultralytics import YOLO
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Load YOLOv8 model
yolo_model = YOLO("best.pt")  # Path to your trained YOLOv8 model

# Class names (as per your data.yaml)
class_names = ['battery', 'can', 'cardboard_bowl', 'cardboard_box', 'chemical_plastic_bottle', 'chemical_plastic_gallon', 
               'chemical_spray_can', 'light_bulb', 'paint_bucket', 'plastic_bag', 'plastic_bottle', 'plastic_bottle_cap', 
               'plastic_box', 'plastic_cultery', 'plastic_cup', 'plastic_cup_lid', 'reuseable_paper', 'scrap_paper', 
               'scrap_plastic', 'snack_bag', 'stick', 'straw']

# Paths
input_image_folder = r"train\images"  # Absolute path to your training images
output_image_folder = "cnn_dataset"  # Folder where images will be saved for CNN training (no cropping)

# -------------------------------------------------------
# Prepare Image Folder Structure
def prepare_data_for_training():
    # Create folder structure for the CNN training
    for class_name in class_names:
        os.makedirs(os.path.join(output_image_folder, class_name), exist_ok=True)

# -------------------------------------------------------
# Function to Save Detected Objects (Without Cropping)
def save_detected_images(image_path, output_dir, class_names):
    # Read the image
    img = cv2.imread(image_path)
    
    # Get detection results from YOLOv8
    results = yolo_model(img)

    # Iterate over each detected object in results
    for box in results[0].boxes:
        # Extract bounding box coordinates, confidence, and class ID
        xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
        conf = box.conf[0].cpu().numpy()  # Confidence score
        cls = int(box.cls[0].cpu().numpy())  # Class ID
        
        # Get the class name using the class ID
        class_name = class_names[cls]
        
        # Save the entire image (without cropping) to the appropriate class folder
        class_folder = os.path.join(output_dir, class_name)
        crop_filename = os.path.join(class_folder, f"{os.path.basename(image_path).split('.')[0]}_detected.jpg")
        
        cv2.imwrite(crop_filename, img)

# -------------------------------------------------------
# Process All Images for Saving (No Cropping)
def process_images():
    for filename in os.listdir(input_image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_image_folder, filename)
            save_detected_images(image_path, output_image_folder, class_names)

# -------------------------------------------------------
# Define the Custom CNN for Classification


class WasteClassifierCNN(nn.Module):
    def __init__(self, num_classes=22):
        super(WasteClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        # 224x224 input → 112x112 → 56x56 after pooling
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # → 112x112
        x = self.pool(F.relu(self.conv2(x)))  # → 56x56
        x = x.view(-1, 64 * 56 * 56)          # Flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# -------------------------------------------------------
# Prepare Data for Training (No Cropping or Transformation)
prepare_data_for_training()

# -------------------------------------------------------
# Define Image Transformations (64x64 for faster training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Must match what your CNN expects
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Create DataLoader for CNN Training
train_data = datasets.ImageFolder(output_image_folder, transform=transform)  # Dataset from the prepared folders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)  # Increased batch size for GPU utilization

# -------------------------------------------------------
# Initialize Model, Loss Function, and Optimizer
model = WasteClassifierCNN(num_classes=22)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------------------------------
# Training with Early Stopping
num_epochs = 10  # Set the number of epochs
patience = 3  # Early stopping patience (3 epochs without improvement)
best_loss = float('inf')
epochs_without_improvement = 0

for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Reset gradients

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Early stopping condition
    if avg_loss < best_loss:
        best_loss = avg_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print("Early stopping")
        break

# Save the trained model
torch.save(model.state_dict(), "waste_cnn_model.pth")
