import torch
import cv2
from ultralytics import YOLO
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import numpy as np
import time
import tkinter as tk
from tkinter import Label

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

# --- Hazardous object classes ---
hazard_classes = ['chemical_plastic_bottle', 'chemical_spray_can', 'paint_bucket']

# --- CNN input transformation ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Webcam setup ---
cap = cv2.VideoCapture(0)
prev_time = time.time()

# --- Tkinter GUI for Header ---
root = tk.Tk()
root.title("Real-Time Hazard Detection")
root.geometry("800x100")  # Header size

# Status label
status_label = Label(root, text="Model Status: Ready", font=("Helvetica", 14), fg="green")
status_label.pack(pady=10)

# FPS label
fps_label = Label(root, text="FPS: 0.00", font=("Helvetica", 14), fg="blue")
fps_label.pack(pady=10)

# Hazard detection status label
hazard_label = Label(root, text="Hazard Alert: None", font=("Helvetica", 14), fg="blue")
hazard_label.pack(pady=10)

root.after(100, lambda: None)  # Keeps the GUI running in the background

def update_gui(hazard_detected, fps):
    # Update the hazard detection status
    if hazard_detected:
        hazard_label.config(text="Hazard Alert: Detected", fg="red")
    else:
        hazard_label.config(text="Hazard Alert: None", fg="blue")
    
    # Update FPS label
    fps_label.config(text=f"FPS: {fps:.2f}")

# --- Title Screen Function ---
def show_title_screen():
    # Create a blank image for the title screen
    title_screen = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Set background color (blue gradient, can customize further)
    for i in range(title_screen.shape[0]):
        color = int(255 * (i / title_screen.shape[0]))  # Gradient effect
        title_screen[i, :] = (color, color, 255)  # RGB (light blue gradient)

    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(title_screen, "Real-Time Hazard Detection", (200, 300),
                font, 2, (255, 255, 255), 5, cv2.LINE_AA)

    # Display title screen
    cv2.imshow("Title Screen", title_screen)

    # Wait for a few seconds (3 seconds here)
    cv2.waitKey(3000)  # 3000 ms (3 seconds)

    cv2.destroyWindow("Title Screen")

# --- Main Loop with Title Screen ---
show_title_screen()  # Show title screen before starting the detection

# Now, start the webcam feed and hazard detection
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Run YOLOv8 (inference on GPU) ---
    results = yolo_model(frame, imgsz=640, verbose=False)[0]  # Reduce imgsz for speed

    hazard_detected = False  # Flag to indicate if any hazard class is detected

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        yolo_class_id = int(box.cls[0])
        yolo_label = class_names[yolo_class_id]
        yolo_conf = float(box.conf[0])

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
            cnn_conf = torch.nn.functional.softmax(cnn_output, dim=1).max().item()  # CNN confidence score

        # Combine YOLO and CNN results
        combined_label = f"YOLO: {yolo_label} ({yolo_conf:.2f}) | CNN: {cnn_label} ({cnn_conf:.2f})"

        # Check if any hazardous object is detected
        if yolo_label in hazard_classes or cnn_label in hazard_classes:
            hazard_detected = True

        # --- Annotate frame ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, combined_label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # --- Add hazard alert overlay if any hazardous object is detected ---
    if hazard_detected:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (0, 0, 255), -1)  # Red background
        cv2.putText(overlay, "HAZARD ALERT", (frame.shape[1]//2 - 150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)  # Apply overlay with transparency

    # --- FPS calculation ---
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time
    
    # Update GUI with FPS and hazard detection status
    update_gui(hazard_detected, fps)

    # --- Display ---
    cv2.imshow("YOLOv8 + CNN Real-Time", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
root.quit()  # Close the Tkinter GUI window
