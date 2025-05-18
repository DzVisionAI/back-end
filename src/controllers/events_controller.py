from flask import Blueprint, jsonify
from src.models.events_model import Event
from src import db

events_bp = Blueprint('events', __name__)

@events_bp.route('/', methods=['GET'], strict_slashes=False)
def get_events():
    events = Event.query.all()
    data = [
        {
            'id': e.id,
            'cameraId': e.cameraId,
            'plateId': e.plateId,
            'time': e.time,
            'typeName': e.typeName,
            'description': e.description,
            'driverId': e.driverId
        } for e in events
    ]
    return jsonify({'success': True, 'data': data}), 200

@events_bp.route('/<int:event_id>', methods=['GET'], strict_slashes=False)
def get_event(event_id):
    e = Event.query.get(event_id)
    if not e:
        return jsonify({'success': False, 'message': 'Event not found'}), 404
    data = {
        'id': e.id,
        'cameraId': e.cameraId,
        'plateId': e.plateId,
        'time': e.time,
        'typeName': e.typeName,
        'description': e.description,
        'driverId': e.driverId
    }
    return jsonify({'success': True, 'data': data}), 200 