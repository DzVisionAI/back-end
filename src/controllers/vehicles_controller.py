from flask import Blueprint, jsonify
from src.models.vehicles_model import Vehicle
from src import db

vehicles_bp = Blueprint('vehicles', __name__)

@vehicles_bp.route('/', methods=['GET'])
def get_vehicles():
    vehicles = Vehicle.query.all()
    data = [
        {
            'id': v.id,
            'plateNumber': v.license_plate.plateNumber if v.license_plate else None,
            'color': v.color,
            'make': v.make,
            'model': v.model,
            'ownerId': v.ownerId,
            'registerAt': v.registerAt,
            'image': v.image
        } for v in vehicles
    ]
    return jsonify({'success': True, 'data': data}), 200

@vehicles_bp.route('/<int:vehicle_id>', methods=['GET'])
def get_vehicle(vehicle_id):
    v = Vehicle.query.get(vehicle_id)
    if not v:
        return jsonify({'success': False, 'message': 'Vehicle not found'}), 404
    data = {
        'id': v.id,
        'plateNumber': v.license_plate.plateNumber if v.license_plate else None,
        'color': v.color,
        'make': v.make,
        'model': v.model,
        'ownerId': v.ownerId,
        'registerAt': v.registerAt,
        'image': v.image
    }
    return jsonify({'success': True, 'data': data}), 200 