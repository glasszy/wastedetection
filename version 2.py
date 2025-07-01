from ultralytics import YOLO
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Load YOLOv8 object detector
yolo_model = YOLO("best.pt")

# Load your custom CNN classifier (PyTorch example)
class WasteCNN(torch.nn.Module):
    def __init__(self):
        super(WasteCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc = torch.nn.Linear(16*64*64, 4)  # 4 classes (adjust as needed)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = x.view(-1, 16*64*64)
        x = self.fc(x)
        return x

cnn_model = WasteCNN()
cnn_model.load_state_dict(torch.load("path_to_cnn_model.pth"))
cnn_model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Open webcam or video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)

    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        cropped = frame[y1:y2, x1:x2]

        # Convert to PIL Image
        pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_img).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            pred = cnn_model(input_tensor)
            class_id = torch.argmax(pred).item()

        label = f"Class {class_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow("Waste Detection + CNN Classification", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
