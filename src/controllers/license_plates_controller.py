from flask import Blueprint, jsonify, request
from src.models.license_plates_model import LicensePlate
from src import db
from src.utils import generate_gcs_signed_url
import os

license_plates_bp = Blueprint('license_plates', __name__)

@license_plates_bp.route('/', methods=['GET'], strict_slashes=False)
def get_license_plates():
    query = LicensePlate.query

    # Query filters
    plate_number = request.args.get('plateNumber')
    camera_id = request.args.get('cameraId')
    detected_at = request.args.get('detectedAt')
    vehicle_id = request.args.get('vehicleId')

    if plate_number:
        query = query.filter(LicensePlate.plateNumber == plate_number)
    if camera_id:
        query = query.filter(LicensePlate.cameraId == camera_id)
    if detected_at:
        query = query.filter(LicensePlate.detectedAt == detected_at)
    # vehicle_id filter is not directly applicable, but we can filter by related vehicle
    if vehicle_id:
        query = query.join(LicensePlate.vehicle).filter_by(id=vehicle_id)

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
    total = query.count()
    plates = query.offset((page - 1) * limit).limit(limit).all()

    bucket_name = os.environ.get('GCS_BUCKET_NAME')
    def get_signed_url(image_url):
        if image_url and bucket_name and image_url.startswith(f'https://storage.googleapis.com/{bucket_name}/'):
            blob_name = image_url.split(f'https://storage.googleapis.com/{bucket_name}/')[-1]
            return generate_gcs_signed_url(blob_name)
        return None
    data = [
        {
            'id': p.id,
            'cameraId': p.cameraId,
            'plateNumber': p.plateNumber,
            'detectedAt': p.detectedAt,
            'image': p.image,
            'signed_url': get_signed_url(p.image),
            'vehicle': {
                'id': p.vehicle.id,
                'color': p.vehicle.color,
                'make': p.vehicle.make,
                'model': p.vehicle.model,
                'ownerId': p.vehicle.ownerId,
                'registerAt': p.vehicle.registerAt,
                'image': p.vehicle.image,
                'signed_url': get_signed_url(p.vehicle.image)
            } if p.vehicle else None
        } for p in plates
    ]
    pagination = {
        'page': page,
        'limit': limit,
        'total': total,
        'pages': (total + limit - 1) // limit
    }
    return jsonify({'success': True, 'data': data, 'pagination': pagination}), 200

@license_plates_bp.route('/<int:plate_id>', methods=['GET'], strict_slashes=False)
def get_license_plate(plate_id):
    p = LicensePlate.query.get(plate_id)
    if not p:
        return jsonify({'success': False, 'message': 'License plate not found'}), 404
    bucket_name = os.environ.get('GCS_BUCKET_NAME')
    def get_signed_url(image_url):
        if image_url and bucket_name and image_url.startswith(f'https://storage.googleapis.com/{bucket_name}/'):
            blob_name = image_url.split(f'https://storage.googleapis.com/{bucket_name}/')[-1]
            return generate_gcs_signed_url(blob_name)
        return None
    data = {
        'id': p.id,
        'cameraId': p.cameraId,
        'plateNumber': p.plateNumber,
        'detectedAt': p.detectedAt,
        'image': p.image,
        'signed_url': get_signed_url(p.image),
        'vehicle': {
            'id': p.vehicle.id,
            'color': p.vehicle.color,
            'make': p.vehicle.make,
            'model': p.vehicle.model,
            'ownerId': p.vehicle.ownerId,
            'registerAt': p.vehicle.registerAt,
            'image': p.vehicle.image,
            'signed_url': get_signed_url(p.vehicle.image)
        } if p.vehicle else None
    }
    return jsonify({'success': True, 'data': data}), 200 