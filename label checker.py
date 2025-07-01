from ultralytics import YOLO
import cv2

# Load the trained YOLOv8 model
model = YOLO('best.pt')  # Make sure best.pt is in the same directory or give full path

print(model.names)