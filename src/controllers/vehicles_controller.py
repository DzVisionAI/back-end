from flask import Blueprint, jsonify, request
from src.models.vehicles_model import Vehicle
from src import db
from src.utils import generate_gcs_signed_url
import os

vehicles_bp = Blueprint('vehicles', __name__)

@vehicles_bp.route('/', methods=['GET'], strict_slashes=False)
def get_vehicles():
    vehicles_query = Vehicle.query.order_by(Vehicle.registerAt.desc())
    # Pagination
    try:
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 10))
        if page < 1:
            page = 1
        if limit < 1:
            limit = 10
    except ValueError:
        page = 1
        limit = 10
    total = vehicles_query.count()
    vehicles = vehicles_query.offset((page - 1) * limit).limit(limit).all()
    bucket_name = os.environ.get('GCS_BUCKET_NAME')
    def get_signed_url(image_url):
        if image_url and bucket_name and image_url.startswith(f'https://storage.googleapis.com/{bucket_name}/'):
            blob_name = image_url.split(f'https://storage.googleapis.com/{bucket_name}/')[-1]
            return generate_gcs_signed_url(blob_name)
        return None
    data = [
        {
            'id': v.id,
            'plateNumber': v.license_plate.plateNumber if v.license_plate else None,
            'color': v.color,
            'make': v.make,
            'model': v.model,
            'ownerId': v.ownerId,
            'registerAt': v.registerAt,
            'image': v.image,
            'signed_url': get_signed_url(v.image)
        } for v in vehicles
    ]
    pagination = {
        'page': page,
        'limit': limit,
        'total': total,
        'pages': (total + limit - 1) // limit
    }
    return jsonify({'success': True, 'data': data, 'pagination': pagination}), 200

@vehicles_bp.route('/<int:vehicle_id>', methods=['GET'], strict_slashes=False)
def get_vehicle(vehicle_id):
    v = Vehicle.query.get(vehicle_id)
    if not v:
        return jsonify({'success': False, 'message': 'Vehicle not found'}), 404
    bucket_name = os.environ.get('GCS_BUCKET_NAME')
    def get_signed_url(image_url):
        if image_url and bucket_name and image_url.startswith(f'https://storage.googleapis.com/{bucket_name}/'):
            blob_name = image_url.split(f'https://storage.googleapis.com/{bucket_name}/')[-1]
            return generate_gcs_signed_url(blob_name)
        return None
    data = {
        'id': v.id,
        'plateNumber': v.license_plate.plateNumber if v.license_plate else None,
        'color': v.color,
        'make': v.make,
        'model': v.model,
        'ownerId': v.ownerId,
        'registerAt': v.registerAt,
        'image': v.image,
        'signed_url': get_signed_url(v.image)
    }
    return jsonify({'success': True, 'data': data}), 200

@vehicles_bp.route('/<int:vehicle_id>', methods=['DELETE'], strict_slashes=False)
def delete_vehicle(vehicle_id):
    v = Vehicle.query.get(vehicle_id)
    if not v:
        return jsonify({'success': False, 'message': 'Vehicle not found'}), 404
    try:
        db.session.delete(v)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Vehicle deleted'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500

@vehicles_bp.route('/<int:vehicle_id>', methods=['PUT'], strict_slashes=False)
def update_vehicle(vehicle_id):
    v = Vehicle.query.get(vehicle_id)
    if not v:
        return jsonify({'success': False, 'message': 'Vehicle not found'}), 404
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'message': 'No data provided'}), 400
    try:
        for field in ['make', 'color', 'model']:
            if field in data:
                setattr(v, field, data[field])
        db.session.commit()
        return jsonify({'success': True, 'message': 'Vehicle updated'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500 