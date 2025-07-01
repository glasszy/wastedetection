import torch
from torchvision import models

# Define your model architecture (e.g., ResNet50)
cnn_model = models.resnet50(pretrained=False)  # Use pretrained=False if you're loading your custom model

# Load the model weights (state_dict)
cnn_model.load_state_dict(torch.load('waste_cnn_model.pth'))

# Set the model to evaluation mode
cnn_model.eval()
