import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO("best.pt")  # Make sure 'best.pt' is in the same directory or provide full path

# Start video capture (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Plot the results on the frame
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
