import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import time
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb
from torch.cuda.amp import autocast, GradScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

# Define paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(CURRENT_DIR, 'config')
COLORS_PATH = os.path.join(CONFIG_DIR, 'colors.json')
MAKES_PATH = os.path.join(CONFIG_DIR, 'makes.json')

# Load configurations
def load_configs():
    """Load color and make configurations"""
    try:
        with open(COLORS_PATH, 'r') as f:
            colors = json.load(f)
        with open(MAKES_PATH, 'r') as f:
            makes = json.load(f)
        return colors, makes
    except Exception as e:
        logging.error(f"Error loading configurations: {str(e)}")
        raise

# Load configurations globally
try:
    COLORS, MAKES = load_configs()
    logging.info(f"Loaded {len(COLORS)} colors and {len(MAKES)} makes from configuration")
except Exception as e:
    logging.error(f"Failed to load configurations: {str(e)}")
    raise

class VehicleDataset(Dataset):
    def __init__(self, image_paths, color_labels, make_labels, transform=None, augment=False):
        self.image_paths = image_paths
        self.color_labels = color_labels
        self.make_labels = make_labels
        self.transform = transform
        self.augment = augment
        
        if augment:
            self.aug_transform = A.Compose([
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
                A.GaussNoise(p=0.3),
                A.RandomShadow(p=0.3),
                A.RandomFog(p=0.2),
                A.RandomSunFlare(p=0.2),
                A.Rotate(limit=15, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.OneOf([
                    A.MotionBlur(p=1),
                    A.GaussianBlur(p=1),
                    A.MedianBlur(p=1)
                ], p=0.3),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = np.array(image)
        
        if self.augment:
            augmented = self.aug_transform(image=image)
            image = augmented['image']
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.color_labels[idx], self.make_labels[idx]

def load_dataset(data_dir, color_mapping, make_mapping):
    """Load and prepare dataset"""
    image_paths = []
    color_labels = []
    make_labels = []
    
    # Walk through dataset directory
    for make in os.listdir(data_dir):
        make_path = os.path.join(data_dir, make)
        if not os.path.isdir(make_path):
            continue
            
        make_idx = make_mapping.get(make.lower())
        if make_idx is None:
            continue
            
        for color in os.listdir(make_path):
            color_path = os.path.join(make_path, color)
            if not os.path.isdir(color_path):
                continue
                
            color_idx = color_mapping.get(color.lower())
            if color_idx is None:
                continue
                
            for img_name in os.listdir(color_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(color_path, img_name))
                    color_labels.append(color_idx)
                    make_labels.append(make_idx)
    
    return image_paths, color_labels, make_labels

def create_model(num_colors, num_makes):
    """Create and initialize the model with improvements"""
    # Use EfficientNet as base model
    model = models.efficientnet_b0(pretrained=True)
    
    # Freeze early layers
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False
    
    # Modified head with attention
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
    
    # Replace classifier
    in_features = model._fc.in_features
    model._fc = nn.Identity()
    
    model = nn.Sequential(
        model,
        nn.Sequential(
            AttentionHead(in_features, num_colors),
            AttentionHead(in_features, num_makes)
        )
    )
    
    return model

def train_model(model, train_loader, val_loader, device, num_epochs=50):
    """Train the model with improvements"""
    # Initialize wandb
    wandb.init(project="vehicle-attribute-detection")
    
    # Loss functions with class weights
    color_criterion = nn.CrossEntropyLoss(weight=calculate_class_weights(train_loader, 'color').to(device))
    make_criterion = nn.CrossEntropyLoss(weight=calculate_class_weights(train_loader, 'make').to(device))
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader)
    )
    
    # Mixed precision training
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    best_model_path = 'best_model.pth'
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        color_correct = 0
        make_correct = 0
        total = 0
        
        for batch_idx, (images, color_labels, make_labels) in enumerate(train_loader):
            images = images.to(device)
            color_labels = color_labels.to(device)
            make_labels = make_labels.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast():
                color_outputs, make_outputs = model(images)
                
                # Calculate losses
                color_loss = color_criterion(color_outputs, color_labels)
                make_loss = make_criterion(make_outputs, make_labels)
                loss = color_loss + make_loss
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            
            # Calculate accuracy
            _, color_preds = torch.max(color_outputs, 1)
            _, make_preds = torch.max(make_outputs, 1)
            
            color_correct += (color_preds == color_labels).sum().item()
            make_correct += (make_preds == make_labels).sum().item()
            total += color_labels.size(0)
            
            train_loss += loss.item()
            
            # Log batch metrics
            if batch_idx % 10 == 0:
                wandb.log({
                    'batch_loss': loss.item(),
                    'learning_rate': scheduler.get_last_lr()[0]
                })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_color_correct = 0
        val_make_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, color_labels, make_labels in val_loader:
                images = images.to(device)
                color_labels = color_labels.to(device)
                make_labels = make_labels.to(device)
                
                color_outputs, make_outputs = model(images)
                
                color_loss = color_criterion(color_outputs, color_labels)
                make_loss = make_criterion(make_outputs, make_labels)
                loss = color_loss + make_loss
                
                val_loss += loss.item()
                
                _, color_preds = torch.max(color_outputs, 1)
                _, make_preds = torch.max(make_outputs, 1)
                
                val_color_correct += (color_preds == color_labels).sum().item()
                val_make_correct += (make_preds == make_labels).sum().item()
                val_total += color_labels.size(0)
        
        # Calculate metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        train_color_acc = 100 * color_correct / total
        train_make_acc = 100 * make_correct / total
        val_color_acc = 100 * val_color_correct / val_total
        val_make_acc = 100 * val_make_correct / val_total
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'train_color_accuracy': train_color_acc,
            'train_make_accuracy': train_make_acc,
            'val_color_accuracy': val_color_acc,
            'val_make_accuracy': val_make_acc
        })
        
        # Log progress
        logging.info(f'Epoch {epoch+1}/{num_epochs}:')
        logging.info(f'Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
        logging.info(f'Train Color Acc: {train_color_acc:.2f}%, Val Color Acc: {val_color_acc:.2f}%')
        logging.info(f'Train Make Acc: {train_make_acc:.2f}%, Val Make Acc: {val_make_acc:.2f}%')
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss,
            }, best_model_path)
            logging.info('Saved new best model')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f'Early stopping after {epoch+1} epochs')
                break
    
    wandb.finish()
    return best_model_path

def calculate_class_weights(dataloader, attribute_type):
    """Calculate class weights to handle imbalanced data"""
    label_counts = torch.zeros(
        len(COLORS) if attribute_type == 'color' else len(MAKES)
    )
    
    for _, color_labels, make_labels in dataloader:
        if attribute_type == 'color':
            for label in color_labels:
                label_counts[label] += 1
        else:
            for label in make_labels:
                label_counts[label] += 1
    
    # Calculate weights
    total = label_counts.sum()
    weights = total / (len(label_counts) * label_counts)
    weights = weights / weights.sum()  # Normalize
    
    return weights

def main():
    # Initialize wandb
    wandb.init(project="vehicle-attribute-detection")
    
    # Create label mappings
    color_mapping = {color.lower(): idx for idx, color in enumerate(COLORS)}
    make_mapping = {make.lower(): idx for idx, make in enumerate(MAKES)}
    
    # Data normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Get the root directory and dataset path
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
    data_dir = os.path.join(ROOT_DIR, 'data', 'vehicle_dataset')
    
    logging.info(f"Looking for dataset in: {data_dir}")
    
    # Check if dataset exists and is prepared
    if not os.path.exists(data_dir):
        logging.error(f"Dataset directory not found: {data_dir}")
        logging.info("Please run prepare_dataset.py first to set up the dataset")
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    
    # Load dataset
    try:
        image_paths, color_labels, make_labels = load_dataset(data_dir, color_mapping, make_mapping)
        logging.info(f"Successfully loaded {len(image_paths)} images")
    except Exception as e:
        logging.error(f"Failed to load dataset: {str(e)}")
        raise
    
    if len(image_paths) == 0:
        logging.error("No images found in the dataset")
        logging.info("Please ensure the dataset is properly prepared with images")
        raise ValueError("No images found in the dataset")
    
    # Split dataset
    train_paths, val_paths, train_colors, val_colors, train_makes, val_makes = train_test_split(
        image_paths, color_labels, make_labels, test_size=0.2, random_state=42
    )
    
    logging.info(f"Split dataset into {len(train_paths)} training and {len(val_paths)} validation samples")
    
    # Create data loaders
    train_dataset = VehicleDataset(
        train_paths, train_colors, train_makes,
        transform=train_transform,
        augment=True
    )
    val_dataset = VehicleDataset(
        val_paths, val_colors, val_makes,
        transform=val_transform,
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(len(COLORS), len(MAKES))
    model = model.to(device)
    
    # Log model architecture
    wandb.watch(model)
    
    # Train model
    best_model_path = train_model(model, train_loader, val_loader, device)
    
    # Move best model to weights directory
    weights_dir = Path('weights')
    weights_dir.mkdir(exist_ok=True)
    Path(best_model_path).rename(weights_dir / 'vehicle_attribute_model.pth')
    logging.info(f'Training complete. Best model saved to {weights_dir}/vehicle_attribute_model.pth')

if __name__ == '__main__':
    main() 