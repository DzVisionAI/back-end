"""Vehicle tracking module using SORT."""
import sys
import os
from pathlib import Path
import numpy as np

# Add SORT to path like in main.py
project_root = str(Path(__file__).resolve().parents[2])
sort_path = os.path.join(project_root, "src", "ai", "sort")
if sort_path not in sys.path:
    sys.path.append(sort_path)

from sort import Sort

class VehicleTracker:
    def __init__(self):
        """Initialize the SORT tracker."""
        try:
            self.tracker = Sort()
            print("Vehicle tracker initialized successfully")
        except Exception as e:
            print(f"Error initializing vehicle tracker: {str(e)}")
            raise Exception(f"Failed to initialize vehicle tracker: {str(e)}")
    
    def update(self, detections):
        """Update tracker with new detections and return track IDs."""
        if not detections:
            return np.empty((0, 5))
        return self.tracker.update(np.asarray(detections)) 