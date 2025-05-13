from flask import Blueprint, jsonify
from src.models.license_plates_model import LicensePlate
from src import db

license_plates_bp = Blueprint('license_plates', __name__)

@license_plates_bp.route('/', methods=['GET'])
def get_license_plates():
    plates = LicensePlate.query.all()
    data = [
        {
            'id': p.id,
            'cameraId': p.cameraId,
            'plateNumber': p.plateNumber,
            'detectedAt': p.detectedAt,
            'image': p.image,
            'vehicleId': p.vehicleId
        } for p in plates
    ]
    return jsonify({'success': True, 'data': data}), 200

@license_plates_bp.route('/<int:plate_id>', methods=['GET'])
def get_license_plate(plate_id):
    p = LicensePlate.query.get(plate_id)
    if not p:
        return jsonify({'success': False, 'message': 'License plate not found'}), 404
    data = {
        'id': p.id,
        'cameraId': p.cameraId,
        'plateNumber': p.plateNumber,
        'detectedAt': p.detectedAt,
        'image': p.image,
        'vehicleId': p.vehicleId
    }
    return jsonify({'success': True, 'data': data}), 200 