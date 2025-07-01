from ultralytics import YOLO
import cv2

# Load the trained YOLOv8 model
model = YOLO('best.pt')  # Make sure best.pt is in the same directory or give full path

# Load the image
image_path = 'bottle.jpg'  # Replace with your image file
image = cv2.imread(image_path)

# Run inference
results = model(image)

# Show results
results[0].show()  # Displays the image with detections (uses OpenCV window)

# Optionally, save the results
# results[0].save(filename='result.jpg')  # Saves an image with bounding boxes
