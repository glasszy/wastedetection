import os
import cv2
from ultralytics import YOLO

# Load YOLOv8 model
yolo_model = YOLO("best.pt")  # Path to your trained YOLOv8 model

# Function to crop and save detected objects
def crop_and_save_images(image_path, output_dir, class_names):
    # Read the image
    img = cv2.imread(image_path)
    
    # Get detection results
    results = yolo_model(img)

    # Iterate over each detected object in results
    for box in results[0].boxes:
        # Extract bounding box coordinates, confidence, and class ID
        xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
        conf = box.conf[0].cpu().numpy()  # Confidence score
        cls = int(box.cls[0].cpu().numpy())  # Class ID
        
        # Get the class name using the class ID
        class_name = class_names[cls]
        
        # Crop the detected object
        cropped_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
        
        # Create class folder if it doesn't exist
        class_folder = os.path.join(output_dir, class_name)
        os.makedirs(class_folder, exist_ok=True)
        
        # Save the cropped image
        crop_filename = os.path.join(class_folder, f"{os.path.basename(image_path).split('.')[0]}_crop.jpg")
        cv2.imwrite(crop_filename, cropped_img)

# Class names (as per your data.yaml)
class_names = ['battery', 'can', 'cardboard_bowl', 'cardboard_box', 'chemical_plastic_bottle', 'chemical_plastic_gallon', 
               'chemical_spray_can', 'light_bulb', 'paint_bucket', 'plastic_bag', 'plastic_bottle', 'plastic_bottle_cap', 
               'plastic_box', 'plastic_cultery', 'plastic_cup', 'plastic_cup_lid', 'reuseable_paper', 'scrap_paper', 
               'scrap_plastic', 'snack_bag', 'stick', 'straw']

# Path to the images
input_image_folder = r"train\images"  # Use the correct folder
output_image_folder = "cnn_dataset"  # Folder to save cropped images

# Process all images in the folder
for filename in os.listdir(input_image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(input_image_folder, filename)
        crop_and_save_images(image_path, output_image_folder, class_names)
