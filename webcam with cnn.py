from PIL import Image, ImageTk
import torch
import torch.nn as nn
import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import Label
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from torchvision import models, transforms

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load YOLOv8 model ---
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

# --- Recycling object classes ---
recycling_classes = ['cardboard_bowl', 'cardboard_box', 'plastic_bag', 'plastic_bottle', 'plastic_bottle_cap',
                     'plastic_box', 'plastic_cultery', 'plastic_cup', 'plastic_cup_lid', 'reuseable_paper',
                     'scrap_paper', 'scrap_plastic', 'snack_bag', 'straw']

# --- CNN input transformation ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Initialize lists to store true and predicted labels ---
y_true = []
y_pred = []

# --- Title Screen Function ---
def show_title_screen():
    title_screen = np.zeros((720, 1280, 3), dtype=np.uint8)
    for i in range(title_screen.shape[0]):
        color = int(255 * (i / title_screen.shape[0]))
        title_screen[i, :] = (color, color, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(title_screen, "Real-Time Hazard Detection", (200, 300),
                font, 2, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.imshow("Title Screen", title_screen)
    cv2.waitKey(3000)
    cv2.destroyWindow("Title Screen")

# --- GUI Setup ---
root = tk.Tk()
root.title("Real-Time Hazard Detection")
root.geometry("1000x600")

# Header (status panel)
status_frame = tk.Frame(root)
status_frame.pack(pady=10)

status_label = Label(status_frame, text="Model Status: Ready", font=("Helvetica", 14), fg="green")
status_label.grid(row=0, column=0, padx=10)

fps_label = Label(status_frame, text="FPS: 0.00", font=("Helvetica", 14), fg="blue")
fps_label.grid(row=0, column=1, padx=10)

hazard_label = Label(status_frame, text="Hazard Alert: None", font=("Helvetica", 14), fg="blue")
hazard_label.grid(row=0, column=2, padx=10)

recycling_label = Label(status_frame, text="Recycling Alert: None", font=("Helvetica", 14), fg="blue")
recycling_label.grid(row=0, column=3, padx=10)

camera_status_label = Label(status_frame, text="Camera Feed: Active", font=("Helvetica", 14), fg="green")
camera_status_label.grid(row=0, column=4, padx=10)

# Webcam Display Area
video_label = Label(root)
video_label.pack()

# Webcam setup
cap = cv2.VideoCapture(0)
prev_time = time.time()

def update_gui(hazard_detected, recycling_detected, fps):
    if hazard_detected:
        hazard_label.config(text="Hazard Alert: Detected", fg="red")
    else:
        hazard_label.config(text="Hazard Alert: None", fg="blue")
        
    if recycling_detected:
        recycling_label.config(text="Recycling Alert: Detected", fg="yellow")
    else:
        recycling_label.config(text="Recycling Alert: None", fg="blue")

    fps_label.config(text=f"FPS: {fps:.2f}")

def process_frame():
    global prev_time
    ret, frame = cap.read()
    if not ret:
        return

    results = yolo_model(frame, imgsz=640, verbose=False)[0]
    hazard_detected = False
    recycling_detected = False

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        yolo_class_id = int(box.cls[0])
        yolo_label = class_names[yolo_class_id]
        yolo_conf = float(box.conf[0])

        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        input_tensor = transform(cropped_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            cnn_output = cnn_model(input_tensor)
            cnn_class_id = torch.argmax(cnn_output, dim=1).item()
            cnn_label = class_names[cnn_class_id]
            cnn_conf = torch.nn.functional.softmax(cnn_output, dim=1).max().item()

        combined_label = f"YOLO: {yolo_label} ({yolo_conf:.2f}) | CNN: {cnn_label} ({cnn_conf:.2f})"

        # Hazard Detection Logic
        if yolo_label in hazard_classes or cnn_label in hazard_classes:
            hazard_detected = True
        
        # Recycling Detection Logic
        if yolo_label in recycling_classes or cnn_label in recycling_classes:
            recycling_detected = True

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, combined_label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        y_true.append(yolo_label)
        y_pred.append(cnn_label)

    # Initialize overlay to a copy of the frame
    overlay = frame.copy()

    # Display Hazard Alert
    if hazard_detected:
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (0, 0, 255), -1)
        cv2.putText(overlay, "HAZARD ALERT", (frame.shape[1]//2 - 150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Display Recycling Alert
    if recycling_detected:
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 50), (0, 255, 255), -1)
        cv2.putText(overlay, "RECYCLING ALERT", (frame.shape[1]//2 - 150, 30),  # Adjusted position and font size
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time
    update_gui(hazard_detected, recycling_detected, fps)

    # Convert to Tkinter-compatible format
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(rgb_image)
    imgtk = ImageTk.PhotoImage(image=img_pil)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, process_frame)

# --- Show Title and Start ---
# show_title_screen()
process_frame()
root.mainloop()

# --- After closing GUI: Compute Metrics ---
cap.release()
cv2.destroyAllWindows()

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

# --- Metrics Plotting ---
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy * 100, precision * 100, recall * 100, f1 * 100]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].bar(metrics, values, color=['blue', 'green', 'red', 'orange'])
axes[0, 0].set_xlabel('Metrics')
axes[0, 0].set_ylabel('Percentage')
axes[0, 0].set_title('Model Evaluation Metrics')
axes[0, 0].set_ylim(0, 100)

sns.heatmap(confusion_matrix(y_true, y_pred, labels=class_names),
            annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[0, 1])
axes[0, 1].set_title('Confusion Matrix')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('True')

y_true_bin = label_binarize(y_true, classes=class_names)
y_pred_bin = label_binarize(y_pred, classes=class_names)

for i in range(y_true_bin.shape[1]):
    precision_vals, recall_vals, _ = precision_recall_curve(y_true_bin[:, i], y_pred_bin[:, i])
    axes[1, 0].plot(recall_vals, precision_vals, label=f'{class_names[i]}')

axes[1, 0].set_xlabel('Recall')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].set_title('Precision-Recall Curves')
axes[1, 0].legend(fontsize="x-small", loc='best')
axes[1, 0].grid(True)

axes[1, 1].axis('off')  # Reserved for future use
plt.tight_layout()
plt.show()
