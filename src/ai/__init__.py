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

__all__ = ['LicensePlateDetector', 'VehicleTracker', 'get_car', 'read_license_plate'] 