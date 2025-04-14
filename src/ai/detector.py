"""License plate detector module using YOLO."""
import os
from pathlib import Path
from ultralytics import YOLO
import torch

# Save the original torch.load function
_original_torch_load = torch.load

# Define a new function that forces weights_only=False
def custom_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

# Override torch.load globally
torch.load = custom_torch_load

class LicensePlateDetector:
    def __init__(self):
        """Initialize the license plate detector with YOLO models."""
        try:
            # Initialize COCO model for vehicle detection
            self.coco_model = YOLO('yolov8n.pt')
            
            # Initialize license plate detector
            project_root = str(Path(__file__).resolve().parents[2])
            license_plate_model_path = os.path.join(project_root, 'src', 'ai', 'license_plate_detector.pt')
            
            if os.path.exists(license_plate_model_path):
                print(f"Loading license plate model from {license_plate_model_path}")
                # Load model directly like in main.py
                self.license_plate_detector = YOLO(license_plate_model_path)
            else:
                print(f"License plate model not found at {license_plate_model_path}")
                raise Exception("License plate model not found")
                
            print("License plate detector initialized successfully")
            
        except Exception as e:
            print(f"Error initializing license plate detector: {str(e)}")
            raise Exception(f"Failed to initialize license plate detector: {str(e)}")
    
    def detect_vehicles(self, frame):
        """Detect vehicles in a frame using COCO model."""
        vehicles = [2, 3, 5, 7]  # car, truck, bus, motorcycle
        detections = []
        
        vehicle_detections = self.coco_model(frame)[0]
        for detection in vehicle_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections.append([x1, y1, x2, y2, score])
        
        return detections
    
    def detect_license_plates(self, frame):
        """Detect license plates in a frame."""
        license_plates = self.license_plate_detector(frame)[0]
        return license_plates.boxes.data.tolist() 