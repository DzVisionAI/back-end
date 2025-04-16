import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
import logging
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)

# Define paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(CURRENT_DIR, 'config')
COLORS_PATH = os.path.join(CONFIG_DIR, 'colors.json')
MAKES_PATH = os.path.join(CONFIG_DIR, 'makes.json')
WEIGHTS_PATH = os.path.join(CURRENT_DIR, 'weights', 'vehicle_attribute_model.pth')

def load_model_and_labels():
    """Load the trained model and label mappings"""
    try:
        # Load labels
        with open(COLORS_PATH, 'r') as f:
            colors = json.load(f)
        with open(MAKES_PATH, 'r') as f:
            makes = json.load(f)
        
        logging.info(f"Loaded {len(colors)} colors and {len(makes)} makes from configuration")
        
        # Create model
        num_colors = len(colors)
        num_makes = len(makes)
        model = create_model(num_colors, num_makes)
        
        # Load weights
        if not os.path.exists(WEIGHTS_PATH):
            raise FileNotFoundError(f"Model weights not found at {WEIGHTS_PATH}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        model = model.to(device)
        model.eval()
        
        return model, colors, makes, device
        
    except Exception as e:
        logging.error(f"Error loading model and labels: {str(e)}")
        raise

def preprocess_image(image_path):
    """Preprocess a single image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def predict_attributes(model, image_tensor, device, colors, makes):
    """Make predictions for a single image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        
        # Split outputs
        color_outputs = outputs[:, :len(colors)]
        make_outputs = outputs[:, len(colors):]
        
        # Get probabilities
        color_probs = torch.softmax(color_outputs, dim=1)
        make_probs = torch.softmax(make_outputs, dim=1)
        
        # Get top predictions and confidences
        color_conf, color_idx = torch.max(color_probs, dim=1)
        make_conf, make_idx = torch.max(make_probs, dim=1)
        
        return {
            'color': colors[color_idx.item()],
            'color_confidence': color_conf.item(),
            'make': makes[make_idx.item()],
            'make_confidence': make_conf.item(),
            'color_distribution': color_probs.cpu().numpy()[0],
            'make_distribution': make_probs.cpu().numpy()[0]
        }

def visualize_prediction(image_path, predictions, save_dir='evaluation_results'):
    """Create a visualization of the model's predictions"""
    # Load configurations for visualization
    with open(COLORS_PATH, 'r') as f:
        colors = json.load(f)
    with open(MAKES_PATH, 'r') as f:
        makes = json.load(f)
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Read and resize image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot original image
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot color prediction
    plt.subplot(2, 2, 2)
    color_text = f"Color: {predictions['color']}\nConfidence: {predictions['color_confidence']:.2%}"
    plt.text(0.5, 0.5, color_text, ha='center', va='center', fontsize=12)
    plt.axis('off')
    
    # Plot color distribution
    plt.subplot(2, 2, 3)
    plt.bar(range(len(colors)), predictions['color_distribution'])
    plt.xticks(range(len(colors)), colors, rotation=45)
    plt.title('Color Probability Distribution')
    
    # Plot make distribution
    plt.subplot(2, 2, 4)
    plt.bar(range(len(makes)), predictions['make_distribution'])
    plt.xticks(range(len(makes)), makes, rotation=45)
    plt.title('Make Probability Distribution')
    
    # Adjust layout and save
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f'prediction_{timestamp}.png')
    plt.savefig(save_path)
    plt.close()
    
    return save_path

def evaluate_test_set(test_dir, model, device, colors, makes, confidence_threshold=0.6):
    """Evaluate model on a test set and generate metrics"""
    true_colors = []
    pred_colors = []
    true_makes = []
    pred_makes = []
    
    for make in os.listdir(test_dir):
        make_path = os.path.join(test_dir, make)
        if not os.path.isdir(make_path):
            continue
            
        for color in os.listdir(make_path):
            color_path = os.path.join(make_path, color)
            if not os.path.isdir(color_path):
                continue
                
            for img_name in os.listdir(color_path):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                img_path = os.path.join(color_path, img_name)
                image_tensor = preprocess_image(img_path)
                predictions = predict_attributes(model, image_tensor, device, colors, makes)
                
                if predictions['color_confidence'] >= confidence_threshold:
                    true_colors.append(color.lower())
                    pred_colors.append(predictions['color'].lower())
                
                if predictions['make_confidence'] >= confidence_threshold:
                    true_makes.append(make.lower())
                    pred_makes.append(predictions['make'].lower())
    
    # Generate reports
    results_dir = os.path.join(CURRENT_DIR, 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Color classification report
    color_report = classification_report(true_colors, pred_colors, output_dict=True)
    with open(os.path.join(results_dir, 'color_classification_report.json'), 'w') as f:
        json.dump(color_report, f, indent=2)
    
    # Make classification report
    make_report = classification_report(true_makes, pred_makes, output_dict=True)
    with open(os.path.join(results_dir, 'make_classification_report.json'), 'w') as f:
        json.dump(make_report, f, indent=2)
    
    # Generate confusion matrices
    plt.figure(figsize=(12, 12))
    color_cm = confusion_matrix(true_colors, pred_colors)
    sns.heatmap(color_cm, annot=True, fmt='d', xticklabels=colors, yticklabels=colors)
    plt.title('Color Confusion Matrix')
    plt.savefig(os.path.join(results_dir, 'color_confusion_matrix.png'))
    plt.close()
    
    plt.figure(figsize=(15, 15))
    make_cm = confusion_matrix(true_makes, pred_makes)
    sns.heatmap(make_cm, annot=True, fmt='d', xticklabels=makes, yticklabels=makes)
    plt.title('Make Confusion Matrix')
    plt.savefig(os.path.join(results_dir, 'make_confusion_matrix.png'))
    plt.close()
    
    return {
        'color_accuracy': color_report['accuracy'],
        'make_accuracy': make_report['accuracy']
    }

def main():
    try:
        # Load model and labels
        model, colors, makes, device = load_model_and_labels()
        logging.info("Model and labels loaded successfully")
        
        # Evaluate on test set
        test_dir = os.path.join(os.path.dirname(CURRENT_DIR), 'data', 'vehicle_dataset', 'test')
        if os.path.exists(test_dir):
            metrics = evaluate_test_set(test_dir, model, device, colors, makes)
            logging.info(f"Test Set Evaluation Results:")
            logging.info(f"Color Accuracy: {metrics['color_accuracy']:.2%}")
            logging.info(f"Make Accuracy: {metrics['make_accuracy']:.2%}")
        
        # Interactive testing
        while True:
            image_path = input("\nEnter image path (or 'q' to quit): ")
            if image_path.lower() == 'q':
                break
                
            if not os.path.exists(image_path):
                logging.error(f"Image not found: {image_path}")
                continue
            
            # Process image
            image_tensor = preprocess_image(image_path)
            predictions = predict_attributes(model, image_tensor, device, colors, makes)
            
            # Visualize results
            vis_path = visualize_prediction(image_path, predictions)
            
            # Print results
            print("\nPrediction Results:")
            print(f"Color: {predictions['color']} (Confidence: {predictions['color_confidence']:.2%})")
            print(f"Make: {predictions['make']} (Confidence: {predictions['make_confidence']:.2%})")
            print(f"Visualization saved to: {vis_path}")
    
    except Exception as e:
        logging.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 