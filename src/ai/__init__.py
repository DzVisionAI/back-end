"""
AI package for license plate detection and vehicle tracking.
This package contains modules for:
- License plate detection and reading
- Vehicle tracking
- Vehicle attribute detection (color and make)
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
import torch.nn as nn

# Load configurations
def load_configs():
    """Load color configurations only"""
    config_dir = os.path.join(os.path.dirname(__file__), 'config')
    with open(os.path.join(config_dir, 'colors.json'), 'r') as f:
        colors = json.load(f)
    return colors

# Load pre-trained models and configurations
def load_vehicle_color_model():
    """Load pre-trained model for vehicle color detection only"""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False
    class AttentionHead(nn.Module):
        def __init__(self, in_features, num_classes):
            super().__init__()
            self.attention = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )
            self.fc = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        def forward(self, x):
            attention_weights = torch.sigmoid(self.attention(x))
            weighted_features = x * attention_weights
            return self.fc(weighted_features)
    colors = load_configs()
    in_features = model.classifier[1].in_features
    model.classifier = AttentionHead(in_features, len(colors))
    model_path = os.path.join(os.path.dirname(__file__), 'weights', 'vehicle_color_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model, colors

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

# Initialize the model and configurations globally
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vehicle_model, COLORS = load_vehicle_color_model()
    vehicle_model = vehicle_model.to(device)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
except Exception as e:
    print(f"Error initializing vehicle color detection: {str(e)}")
    raise

def preprocess_vehicle_image(image):
    """Preprocess vehicle image for the model"""
    # Convert BGR to RGB if needed
    if isinstance(image, np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    
    # Apply preprocessing
    image = preprocess(image)
    return image.unsqueeze(0)

def detect_vehicle_color(image):
    """Detect the color of a vehicle from its image"""
    try:
        image_tensor = preprocess_vehicle_image(image).to(device)
        with torch.no_grad():
            output = vehicle_model(image_tensor)
            color_probs = torch.softmax(output, dim=1)
            color_idx = torch.argmax(color_probs, dim=1).item()
            confidence = color_probs[0][color_idx].item()
        if confidence > 0.5:
            return COLORS[color_idx]
        return None
    except Exception as e:
        print(f"Error detecting vehicle color: {str(e)}")
        return None

def detect_vehicle_make(image):
    """Detect the make of a vehicle from its image"""
    try:
        # Preprocess image
        image_tensor = preprocess_vehicle_image(image).to(device)
        
        # Get model predictions
        with torch.no_grad():
            _, make_output = vehicle_model(image_tensor)
            make_probs = torch.softmax(make_output, dim=1)
            make_idx = torch.argmax(make_probs, dim=1).item()
            confidence = make_probs[0][make_idx].item()
        
        # Return make if confidence is high enough
        if confidence > 0.5:  # Adjust threshold as needed
            return MAKES[make_idx]
        return None
        
    except Exception as e:
        print(f"Error detecting vehicle make: {str(e)}")
        return None

__all__ = ['LicensePlateDetector', 'VehicleTracker', 'get_car', 'read_license_plate'] 