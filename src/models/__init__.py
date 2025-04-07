from src.models.user_model import User
from src.models.blacklist_model import BlackList
from src.models.users_permissions_model import UsersPermissions
from src.models.notification_model import Notification
from src.models.alerts_model import Alert
from src.models.events_model import Event
from src.models.license_plates_model import LicensePlate
from src.models.cameras_model import Camera
from src.models.face_recognition_model import FaceRecognition
from src.models.drivers_model import Driver
from src.models.vehicles_model import Vehicle

__all__ = [
    'User',
    'BlackList',
    'UsersPermissions',
    'Notification',
    'Alert',
    'Event',
    'LicensePlate',
    'Camera',
    'FaceRecognition',
    'Driver',
    'Vehicle'
]
