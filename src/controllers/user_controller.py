from flask import Blueprint, request, jsonify
from src.services.user_service import UserService
from src.middlewares.auth_middleware import admin_required

user_bp = Blueprint('user', __name__)

@user_bp.route('/', methods=['GET'])
@admin_required()
def get_users():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    result = UserService.get_all_users(page, per_page)
    return jsonify(result), 200 if result['success'] else 400

@user_bp.route('/<int:user_id>', methods=['GET'])
@admin_required()
def get_user(user_id):
    result = UserService.get_user(user_id)
    return jsonify(result), 200 if result['success'] else 404

@user_bp.route('/', methods=['POST'])
@admin_required()
def create_user():
    data = request.get_json()
    if not data or not all(k in data for k in ('username', 'email', 'password')):
        return jsonify({
            'success': False,
            'message': 'Username, email and password are required'
        }), 400

    result = UserService.create_user(data)
    return jsonify(result), 201 if result['success'] else 400

@user_bp.route('/<int:user_id>', methods=['PUT'])
@admin_required()
def update_user(user_id):
    data = request.get_json()
    if not data:
        return jsonify({
            'success': False,
            'message': 'No data provided'
        }), 400

    result = UserService.update_user(user_id, data)
    return jsonify(result), 200 if result['success'] else 400

@user_bp.route('/<int:user_id>', methods=['DELETE'])
@admin_required()
def delete_user(user_id):
    result = UserService.delete_user(user_id)
    return jsonify(result), 200 if result['success'] else 404 