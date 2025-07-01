import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from PIL import Image
import os

# 1. Define the CNN model
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

# 2. Define image transformations (resize, normalization)
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to 64x64 for faster training/inference
    transforms.ToTensor(),
])

# 3. Prepare data loaders (train and test)
train_image_folder = 'cnn_dataset'  # Replace with actual train data path
test_image_folder = 'cnn_dataset'    # Replace with actual test data path

# Create datasets
train_data = datasets.ImageFolder(train_image_folder, transform=transform)
test_data = datasets.ImageFolder(test_image_folder, transform=transform)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 4. Initialize the model, loss function, and optimizer
model = WasteClassifierCNN(num_classes=22)  # Initialize the model with 22 classes
criterion = nn.CrossEntropyLoss()  # Loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer

# 5. Training loop
num_epochs = 10  # Set number of epochs
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

# 6. Save the trained model weights (recommended)
torch.save(model.state_dict(), "final_waste_cnn_model.pth")

# 7. Evaluate the model
correct = 0
total = 0
model.eval()  # Set to evaluation mode
with torch.no_grad():  # Disable gradient calculation for evaluation
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy on test dataset: {accuracy:.2f}%")

# 8. Load the saved model weights (for inference)
# To load the model later:
model = WasteClassifierCNN(num_classes=22)  # Reinitialize the model
model.load_state_dict(torch.load("final_waste_cnn_model.pth"))  # Load the weights
model.eval()  # Set to evaluation mode

# 9. Inference: Predict a new image
def predict(image_path):
    img = Image.open(image_path)  # Open the image
    img = transform(img)  # Apply the same transformation as during training
    img = img.unsqueeze(0)  # Add batch dimension
    
    # Perform inference
    with torch.no_grad():
        outputs = model(img)
        _, predicted_class = torch.max(outputs, 1)
    
    # Map predicted class index to class name
    class_names = ['battery', 'can', 'cardboard_bowl', 'cardboard_box', 'chemical_plastic_bottle', 
                   'chemical_plastic_gallon', 'chemical_spray_can', 'light_bulb', 'paint_bucket', 
                   'plastic_bag', 'plastic_bottle', 'plastic_bottle_cap', 'plastic_box', 'plastic_cultery', 
                   'plastic_cup', 'plastic_cup_lid', 'reuseable_paper', 'scrap_paper', 'scrap_plastic', 
                   'snack_bag', 'stick', 'straw']
    predicted_class_name = class_names[predicted_class.item()]
    print(f"Predicted class: {predicted_class_name}")

# Test with a new image
image_path = 'bottle.jpg'  # Replace with your new image path
predict(image_path)
