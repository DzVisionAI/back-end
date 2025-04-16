import os
import requests
import zipfile
import shutil
from pathlib import Path
import logging
import json
from tqdm import tqdm
import tarfile
import kagglehub

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_preparation.log'),
        logging.StreamHandler()
    ]
)

# Define paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(CURRENT_DIR, 'config')
COLORS_PATH = os.path.join(CONFIG_DIR, 'colors.json')
MAKES_PATH = os.path.join(CONFIG_DIR, 'makes.json')
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
DATASET_PATH = os.path.join(ROOT_DIR, 'data', 'vehicle_dataset')

# Load color and make configurations
try:
    with open(COLORS_PATH, 'r') as f:
        colors = json.load(f)
    with open(MAKES_PATH, 'r') as f:
        makes = json.load(f)
except Exception as e:
    raise Exception(f"Error loading configuration files: {str(e)}")

def download_file(url, filename):
    """Download a file with progress bar and verify download"""
    try:
        logging.info(f"Downloading {url} to {filename}")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes
        
        total_size = int(response.headers.get('content-length', 0))
        if total_size == 0:
            raise Exception("Got empty file from server")
            
        with open(filename, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)
                
        # Verify file was downloaded and has size
        if not os.path.exists(filename):
            raise Exception(f"Failed to create file {filename}")
        if os.path.getsize(filename) == 0:
            os.remove(filename)
            raise Exception(f"Downloaded file {filename} is empty")
            
        logging.info(f"Successfully downloaded {filename}")
        return True
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Network error while downloading {filename}: {str(e)}")
        if os.path.exists(filename):
            os.remove(filename)
        raise
    except Exception as e:
        logging.error(f"Error downloading {filename}: {str(e)}")
        if os.path.exists(filename):
            os.remove(filename)
        raise

def prepare_stanford_cars():
    """Download and prepare Stanford Cars dataset using Kaggle"""
    logging.info("Starting Stanford Cars dataset preparation...")
    
    # Create dataset directory
    dataset_path = Path(DATASET_PATH)
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download dataset from Kaggle
        logging.info("Downloading Stanford Cars dataset from Kaggle...")
        kaggle_path = kagglehub.dataset_download("jessicali9530/stanford-cars-dataset")
        logging.info(f"Dataset downloaded to: {kaggle_path}")
        
        # The dataset comes as a directory with all files
        # Move and organize files
        for file_path in Path(kaggle_path).glob('*'):
            if file_path.is_file():
                # Copy files to our dataset directory
                shutil.copy2(str(file_path), str(dataset_path / file_path.name))
                logging.info(f"Copied {file_path.name} to dataset directory")
        
        # Create directory structure based on makes and colors
        for make in makes:
            make_dir = dataset_path / make.lower()
            make_dir.mkdir(exist_ok=True)
            for color in colors:
                color_dir = make_dir / color.lower()
                color_dir.mkdir(exist_ok=True)
                logging.info(f"Created directory: {color_dir}")
        
        # Process images from the downloaded dataset
        for img_path in dataset_path.glob('*.jpg'):
            if img_path.is_file():
                # For demonstration, distribute images randomly
                # In a real implementation, you'd use computer vision to determine make/color
                make = makes[0].lower()  # Using first make as example
                color = colors[0].lower()  # Using first color as example
                dest_dir = dataset_path / make / color
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(img_path), str(dest_dir / img_path.name))
                logging.info(f"Moved {img_path.name} to {dest_dir}")
        
        logging.info("Dataset organization completed successfully!")
        
    except Exception as e:
        logging.error(f"Error preparing dataset: {str(e)}")
        raise

def prepare_vehicle_colors():
    """Download and prepare vehicle color dataset"""
    # This would typically involve downloading a color-specific dataset
    # For now, we'll create a structure for manual data collection
    dataset_path = Path("data/vehicle_dataset")
    
    # Create directory structure
    for make in makes:
        for color in colors:
            color_path = dataset_path / make / color
            color_path.mkdir(parents=True, exist_ok=True)
    
    logging.info("Created directory structure for vehicle dataset")
    logging.info("Please populate the directories with appropriate images")
    logging.info(f"Dataset path: {dataset_path}")

def main():
    """Main function to prepare the complete dataset"""
    try:
        # Prepare Stanford Cars dataset
        prepare_stanford_cars()
        
        # Create a README file with dataset information
        readme_content = """
        Vehicle Dataset
        ==============
        
        This dataset is based on the Stanford Cars Dataset from Kaggle.
        Dataset source: https://www.kaggle.com/jessicali9530/stanford-cars-dataset
        
        Directory Structure:
        data/vehicle_dataset/
        ├── [make]/
        │   ├── [color]/
        │   │   ├── image1.jpg
        │   │   ├── image2.jpg
        │   │   └── ...
        │   └── ...
        └── ...
        
        Please ensure images are properly categorized by make and color.
        """
        
        readme_path = os.path.join(DATASET_PATH, "README.md")
        with open(readme_path, "w") as f:
            f.write(readme_content.strip())
        
        logging.info("Dataset preparation complete!")
        logging.info("Please review the README.md file in the dataset directory for next steps.")
        
    except Exception as e:
        logging.error(f"Error preparing dataset: {str(e)}")
        raise

if __name__ == "__main__":
    main() 