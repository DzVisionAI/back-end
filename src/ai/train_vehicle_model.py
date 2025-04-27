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
import scipy.io
import kagglehub

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

class StanfordCarsDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, augment=False):
        self.image_paths = image_paths
        self.labels = labels
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
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            image = np.array(image)
            
            if self.augment:
                augmented = self.aug_transform(image=image)
                image = augmented['image']
            
            if self.transform:
                image = self.transform(image)
            
            return image, self.labels[idx]
        except Exception as e:
            logging.error(f"Error loading image {self.image_paths[idx]}: {str(e)}")
            # Return a default image or skip this sample
            raise e

def load_stanford_cars_dataset():
    """Load Stanford Cars dataset using kagglehub with improved handling"""
    logging.info("Loading Stanford Cars dataset...")
    
    try:
        # Download dataset using kagglehub
        logging.info("Downloading Stanford Cars dataset from Kaggle...")
        kaggle_paths = kagglehub.dataset_download("jessicali9530/stanford-cars-dataset")
        base_dir = kaggle_paths[0] if isinstance(kaggle_paths, (list, tuple)) else kaggle_paths
        logging.info(f"Dataset downloaded to: {base_dir}")
        
        # Find annotation files
        cars_annos = None
        cars_train_dir = None
        cars_test_dir = None
        
        for root, dirs, files in os.walk(base_dir):
            if "cars_annos.mat" in files:
                cars_annos = os.path.join(root, "cars_annos.mat")
            if "cars_train" in dirs:
                cars_train_dir = os.path.join(root, "cars_train")
            if "cars_test" in dirs:
                cars_test_dir = os.path.join(root, "cars_test")
        
        if not all([cars_annos, cars_train_dir, cars_test_dir]):
            raise FileNotFoundError("Could not find all required dataset files")
            
        logging.info(f"Found annotations at: {cars_annos}")
        logging.info(f"Train directory: {cars_train_dir}")
        logging.info(f"Test directory: {cars_test_dir}")
        
        # Load annotations
        annotations = scipy.io.loadmat(cars_annos)
        
        # Extract class names
        class_names = [name[0] for name in annotations['class_names'][0]]
        logging.info(f"Found {len(class_names)} car classes")
        
        # Process annotations
        image_paths = []
        make_labels = []
        
        for anno in annotations['annotations'][0]:
            img_name = os.path.basename(anno[0][0])  # Get filename only
            class_id = anno[5][0][0] - 1  # class IDs are 1-indexed
            
            # Use cars_test/cars_test directory
            img_dir = os.path.join(cars_test_dir, 'cars_test')
            img_path = os.path.join(img_dir, img_name)
            
            if os.path.exists(img_path):
                try:
                    # Extract make from class name (format: "Make Model Year")
                    class_name = class_names[class_id]
                    make = class_name.split()[0].lower()
                    
                    # Map make to our predefined makes
                    make_idx = None
                    for idx, valid_make in enumerate(MAKES):
                        if make in valid_make.lower():
                            make_idx = idx
                            break
                    
                    if make_idx is not None:
                        image_paths.append(img_path)
                        make_labels.append(make_idx)
                        
                        # Log progress
                        if len(image_paths) % 1000 == 0:
                            logging.info(f"Processed {len(image_paths)} images...")
                            
                except Exception as e:
                    logging.warning(f"Error processing annotation for {img_path}: {str(e)}")
                    continue
            else:
                logging.warning(f"Image not found: {img_path}")
        
        if not image_paths:
            raise ValueError("No valid images found in the dataset")
            
        logging.info(f"Successfully loaded {len(image_paths)} valid images")
        logging.info(f"Found {len(set(make_labels))} unique makes in the dataset")
        
        # Log some example makes for verification
        unique_makes = set(make_labels)
        logging.info(f"Example makes found: {[MAKES[idx] for idx in list(unique_makes)[:10]]}")
        
        return image_paths, make_labels
        
    except Exception as e:
        logging.error(f"Error loading Stanford Cars dataset: {str(e)}")
        raise

def create_model(num_makes):
    """Create and initialize the model"""
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
                nn.Dropout(0.5),  # Increased dropout for better regularization
                nn.Linear(512, num_classes)
            )
        
        def forward(self, x):
            attention_weights = torch.sigmoid(self.attention(x))
            weighted_features = x * attention_weights
            return self.fc(weighted_features)
    
    # Replace classifier
    in_features = model._fc.in_features
    model._fc = AttentionHead(in_features, num_makes)
    
    return model

def train_model(model, train_loader, val_loader, device, num_epochs=50):
    """Train the model with improvements"""
    # Initialize wandb
    wandb.init(project="vehicle-attribute-detection", config={
        "architecture": "EfficientNet-B0",
        "dataset": "Stanford Cars",
        "epochs": num_epochs,
        "batch_size": train_loader.batch_size,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.01
    })
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=calculate_class_weights(train_loader, 'make').to(device))
    
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
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
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
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        train_acc = 100 * correct / total
        val_acc = 100 * val_correct / val_total
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc
        })
        
        # Log progress
        logging.info(f'Epoch {epoch+1}/{num_epochs}:')
        logging.info(f'Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
        logging.info(f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
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
    label_counts = torch.zeros(len(MAKES))
    
    for _, labels in dataloader:
        for label in labels:
            label_counts[label] += 1
    
    # Calculate weights
    total = label_counts.sum()
    weights = total / (len(label_counts) * label_counts)
    weights = weights / weights.sum()  # Normalize
    
    return weights

def main():
    # Initialize wandb
    wandb.init(project="vehicle-attribute-detection")
    
    # Data normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Load Stanford Cars dataset
    try:
        image_paths, make_labels = load_stanford_cars_dataset()
        logging.info(f"Successfully loaded {len(image_paths)} images")
    except Exception as e:
        logging.error(f"Failed to load dataset: {str(e)}")
        raise
    
    # Split dataset
    train_paths, val_paths, train_makes, val_makes = train_test_split(
        image_paths, make_labels, test_size=0.2, random_state=42, stratify=make_labels
    )
    
    logging.info(f"Split dataset into {len(train_paths)} training and {len(val_paths)} validation samples")
    
    # Create data loaders
    train_dataset = StanfordCarsDataset(
        train_paths, train_makes,
        transform=train_transform,
        augment=True
    )
    val_dataset = StanfordCarsDataset(
        val_paths, val_makes,
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
    model = create_model(len(MAKES))
    model = model.to(device)
    
    # Log model architecture
    wandb.watch(model)
    
    # Train model
    best_model_path = train_model(model, train_loader, val_loader, device)
    
    # Move best model to weights directory
    weights_dir = Path('weights')
    weights_dir.mkdir(exist_ok=True)
    Path(best_model_path).rename(weights_dir / 'vehicle_make_model.pth')
    logging.info(f'Training complete. Best model saved to {weights_dir}/vehicle_make_model.pth')

if __name__ == '__main__':
    main() 