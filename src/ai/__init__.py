"""
AI package for license plate detection and vehicle tracking.
This package contains modules for:
- License plate detection and reading
- Vehicle tracking
- Utility functions for processing detections
"""

from .detector import LicensePlateDetector
from .tracker import VehicleTracker
from .util import get_car, read_license_plate

import cv2
import numpy as np
import torch
from torchvision import transforms, models
from PIL import Image
import os
import json
from pathlib import Path

# Load pre-trained models and configurations
def load_vehicle_attribute_model():
    """Load pre-trained ResNet model for vehicle attribute detection"""
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    
    # Modify the final layer for our specific tasks
    # Color classes + Make classes
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(256, 128)
    )
    
    # Load the trained weights if they exist
    model_path = os.path.join(os.path.dirname(__file__), 'weights', 'vehicle_attribute_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    
    model.eval()
    return model

# Load color and make labels
def load_labels():
    """Load color and make labels from JSON configuration"""
    config_path = os.path.join(os.path.dirname(__file__), 'config')
    
    with open(os.path.join(config_path, 'colors.json'), 'r') as f:
        colors = json.load(f)
    
    with open(os.path.join(config_path, 'makes.json'), 'r') as f:
        makes = json.load(f)
    
    return colors, makes

# Initialize the model and labels globally
try:
    vehicle_model = load_vehicle_attribute_model()
    color_labels, make_labels = load_labels()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vehicle_model = vehicle_model.to(device)
    
    # Define image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
except Exception as e:
    print(f"Error initializing vehicle attribute detection: {str(e)}")
    raise

def preprocess_vehicle_image(image):
    """Preprocess vehicle image for the model"""
    # Convert BGR to RGB
    if isinstance(image, np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    
    # Apply preprocessing
    image = preprocess(image)
    return image.unsqueeze(0)

def detect_vehicle_color(image):
    """
    Detect the color of a vehicle in the image.
    
    Args:
        image: numpy array (BGR format) containing the vehicle image
        
    Returns:
        str: Detected color name
    """
    try:
        # Preprocess image
        input_tensor = preprocess_vehicle_image(image)
        input_tensor = input_tensor.to(device)
        
        # Get model prediction
        with torch.no_grad():
            features = vehicle_model(input_tensor)
            color_features = features[:, :len(color_labels)]
            color_probs = torch.softmax(color_features, dim=1)
            color_idx = torch.argmax(color_probs).item()
        
        # Get color name with confidence
        color_name = color_labels[color_idx]
        confidence = color_probs[0][color_idx].item()
        
        # Only return color if confidence is high enough
        if confidence > 0.6:  # Confidence threshold
            return color_name
        return None
        
    except Exception as e:
        print(f"Error in color detection: {str(e)}")
        return None

def detect_vehicle_make(image):
    """
    Detect the make of a vehicle in the image.
    
    Args:
        image: numpy array (BGR format) containing the vehicle image
        
    Returns:
        str: Detected vehicle make
    """
    try:
        # Preprocess image
        input_tensor = preprocess_vehicle_image(image)
        input_tensor = input_tensor.to(device)
        
        # Get model prediction
        with torch.no_grad():
            features = vehicle_model(input_tensor)
            make_features = features[:, len(color_labels):]
            make_probs = torch.softmax(make_features, dim=1)
            make_idx = torch.argmax(make_probs).item()
        
        # Get make name with confidence
        make_name = make_labels[make_idx]
        confidence = make_probs[0][make_idx].item()
        
        # Only return make if confidence is high enough
        if confidence > 0.7:  # Higher threshold for make detection
            return make_name
        return None
        
    except Exception as e:
        print(f"Error in make detection: {str(e)}")
        return None

# Create necessary directories and configuration files
def initialize_ai_files():
    """Initialize necessary directories and configuration files"""
    try:
        # Create directory structure
        base_dir = os.path.dirname(__file__)
        Path(os.path.join(base_dir, 'weights')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(base_dir, 'config')).mkdir(parents=True, exist_ok=True)
        
        # Create color configuration if it doesn't exist
        colors_path = os.path.join(base_dir, 'config', 'colors.json')
        if not os.path.exists(colors_path):
            colors = [
                "black", "white", "silver", "gray", "red",
                "blue", "brown", "green", "beige", "gold",
                "yellow", "orange", "purple", "pink"
            ]
            with open(colors_path, 'w') as f:
                json.dump(colors, f, indent=2)
        
        # Create makes configuration if it doesn't exist
        makes_path = os.path.join(base_dir, 'config', 'makes.json')
        if not os.path.exists(makes_path):
            makes = [
                "Toyota", "Honda", "Ford", "Chevrolet", "Volkswagen",
                "BMW", "Mercedes-Benz", "Audi", "Hyundai", "Kia",
                "Nissan", "Mazda", "Subaru", "Lexus", "Porsche",
                "Ferrari", "Lamborghini", "Tesla", "Volvo", "Land Rover"
            ]
            with open(makes_path, 'w') as f:
                json.dump(makes, f, indent=2)
                
    except Exception as e:
        print(f"Error initializing AI files: {str(e)}")
        raise

# Initialize files when module is imported
initialize_ai_files()

__all__ = ['LicensePlateDetector', 'VehicleTracker', 'get_car', 'read_license_plate'] 