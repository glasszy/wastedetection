from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transforms for data augmentation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Create ImageFolder dataset
train_data = datasets.ImageFolder("cnn_dataset", transform=transform)

# Create DataLoader for batch processing
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
