from flask import Blueprint, jsonify
from src.models.drivers_model import Driver
from src import db

drivers_bp = Blueprint('drivers', __name__)

@drivers_bp.route('/', methods=['GET'])
def get_drivers():
    drivers = Driver.query.all()
    data = [
        {
            'id': d.id,
            'cameraId': d.cameraId,
            'vehicleId': d.vehicleId,
            'behaviourType': d.behaviourType,
            'image': d.image,
            'detectedAt': d.detectedAt
        } for d in drivers
    ]
    return jsonify({'success': True, 'data': data}), 200

@drivers_bp.route('/<int:driver_id>', methods=['GET'])
def get_driver(driver_id):
    d = Driver.query.get(driver_id)
    if not d:
        return jsonify({'success': False, 'message': 'Driver not found'}), 404
    data = {
        'id': d.id,
        'cameraId': d.cameraId,
        'vehicleId': d.vehicleId,
        'behaviourType': d.behaviourType,
        'image': d.image,
        'detectedAt': d.detectedAt
    }
    return jsonify({'success': True, 'data': data}), 200 