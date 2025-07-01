import torch
import cv2
from ultralytics import YOLO
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load YOLOv8 model (GPU by default if available) ---
yolo_model = YOLO("best.pt")
yolo_model.to(device)

# --- Load CNN model ---
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

cnn_model = CNNModel(num_classes=22)
cnn_model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
cnn_model.to(device)
cnn_model.eval()

# --- Class names from data.yaml ---
class_names = ['battery', 'can', 'cardboard_bowl', 'cardboard_box', 'chemical_plastic_bottle',
               'chemical_plastic_gallon', 'chemical_spray_can', 'light_bulb', 'paint_bucket',
               'plastic_bag', 'plastic_bottle', 'plastic_bottle_cap', 'plastic_box',
               'plastic_cultery', 'plastic_cup', 'plastic_cup_lid', 'reuseable_paper',
               'scrap_paper', 'scrap_plastic', 'snack_bag', 'stick', 'straw']

# --- Hazardous object class names --- (can be customized as needed)
hazardous_classes = ['chemical_plastic_bottle', 'chemical_plastic_gallon', 'chemical_spray_can']

# --- CNN input transformation ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- GUI to upload an image using tkinter ---
def upload_image():
    # Open file dialog to select an image
    image_path = filedialog.askopenfilename(title="Select an Image", 
                                            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])
    if not image_path:
        return  # Exit if no file is selected

    # Read image using OpenCV
    frame = cv2.imread(image_path)

    if frame is None:
        print("Error: Image not found.")
        return

    prev_time = time.time()

    # --- Run YOLOv8 (inference on GPU) ---
    results = yolo_model(frame, imgsz=640, verbose=False)[0]  # Reduce imgsz for speed

    hazard_detected = False

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        yolo_class_id = int(box.cls[0])
        yolo_label = class_names[yolo_class_id]
        conf = float(box.conf[0])

        # --- Crop and classify with CNN ---
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            continue  # Skip bad crops

        # Convert for CNN (to RGB -> PIL -> Tensor)
        cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        input_tensor = transform(cropped_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            cnn_output = cnn_model(input_tensor)
            cnn_class_id = torch.argmax(cnn_output, dim=1).item()
            cnn_label = class_names[cnn_class_id]

        # --- Check if the object is hazardous ---
        is_hazardous = cnn_label in hazardous_classes

        # --- Combine YOLO and CNN result into one annotation ---
        label_text = f"YOLO: {yolo_label} ({conf:.2f}) | CNN: {cnn_label} | {'HAZARD!' if is_hazardous else 'Safe'}"

        # --- Draw bounding box and annotations ---
        color = (0, 255, 0) if not is_hazardous else (0, 0, 255)
        thickness = 3 if is_hazardous else 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- Show warning if hazardous object detected ---
        if is_hazardous:
            cv2.putText(frame, "WARNING: HAZARD DETECTED!", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # --- Display the result ---
    cv2.imshow("YOLOv8 + CNN Inference", frame)

    # Wait until user closes the window
    cv2.waitKey(0)

    # Cleanup
    cv2.destroyAllWindows()

# --- Set up the Tkinter root window ---
root = tk.Tk()
root.withdraw()  # Hide the main tkinter window

# --- Open the file dialog to upload an image ---
upload_image()

# --- Close the Tkinter window after processing ---
root.quit()  # This will close the Tkinter event loop and exit the program.
