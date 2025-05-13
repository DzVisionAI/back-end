from flask import Blueprint, request, jsonify
from flask_jwt_extended import get_jwt_identity, verify_jwt_in_request
from src.models.blacklist_model import BlackListSchema
from src.services.blacklist_service import BlackListService
from src.middlewares.auth_middleware import admin_required

blacklist_bp = Blueprint('blacklist', __name__)
schema = BlackListSchema()

@blacklist_bp.route('/', methods=['GET'])
def get_blacklists():
    result = BlackListService.get_all()
    return jsonify(result), 200 if result['success'] else 400

@blacklist_bp.route('/<int:blacklist_id>', methods=['GET'])
def get_blacklist(blacklist_id):
    result = BlackListService.get_by_id(blacklist_id)
    return jsonify(result), 200 if result['success'] else 404

@blacklist_bp.route('/', methods=['POST'])
@admin_required()
def create_blacklist():
    data = request.get_json()
    errors = schema.validate(data)
    if errors:
        return jsonify({'success': False, 'errors': errors}), 400
    verify_jwt_in_request()
    user_id = get_jwt_identity()
    data['addedBy'] = user_id
    result = BlackListService.create(data)
    return jsonify(result), 201 if result['success'] else 400

@blacklist_bp.route('/<int:blacklist_id>', methods=['PUT'])
@admin_required()
def update_blacklist(blacklist_id):
    data = request.get_json()
    errors = schema.validate(data, partial=True)
    if errors:
        return jsonify({'success': False, 'errors': errors}), 400
    verify_jwt_in_request()
    user_id = get_jwt_identity()
    data['addedBy'] = user_id  # Optionally track who updated
    result = BlackListService.update(blacklist_id, data)
    return jsonify(result), 200 if result['success'] else 400

@blacklist_bp.route('/<int:blacklist_id>', methods=['DELETE'])
@admin_required()
def delete_blacklist(blacklist_id):
    verify_jwt_in_request()
    user_id = get_jwt_identity()
    result = BlackListService.delete(blacklist_id, user_id)
    return jsonify(result), 200 if result['success'] else 404 