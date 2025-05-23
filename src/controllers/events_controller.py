from flask import Blueprint, jsonify, request
from src.models.events_model import Event
from src import db

events_bp = Blueprint('events', __name__)

@events_bp.route('/', methods=['GET'], strict_slashes=False)
def get_events():
    events_query = Event.query
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
    total = events_query.count()
    events = events_query.offset((page - 1) * limit).limit(limit).all()
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
    pagination = {
        'page': page,
        'limit': limit,
        'total': total,
        'pages': (total + limit - 1) // limit
    }
    return jsonify({'success': True, 'data': data, 'pagination': pagination}), 200

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