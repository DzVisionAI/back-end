from flask import Blueprint, jsonify
from src.models.cameras_model import Camera
from src import db

cameras_bp = Blueprint('cameras', __name__)

@cameras_bp.route('/', methods=['GET'], strict_slashes=False)
def get_cameras():
    cameras = Camera.query.all()
    data = [
        {
            'id': c.id,
            'name': c.name,
            'type': c.type,
            'ipAddress': c.ipAddress,
            'macAddress': c.macAddress,
            'status': c.status,
            'location': c.location,
            'typeName': c.typeName,
            'description': c.description
        } for c in cameras
    ]
    return jsonify({'success': True, 'data': data}), 200

@cameras_bp.route('/<int:camera_id>', methods=['GET'], strict_slashes=False)
def get_camera(camera_id):
    c = Camera.query.get(camera_id)
    if not c:
        return jsonify({'success': False, 'message': 'Camera not found'}), 404
    data = {
        'id': c.id,
        'name': c.name,
        'type': c.type,
        'ipAddress': c.ipAddress,
        'macAddress': c.macAddress,
        'status': c.status,
        'location': c.location,
        'typeName': c.typeName,
        'description': c.description
    }
    return jsonify({'success': True, 'data': data}), 200 