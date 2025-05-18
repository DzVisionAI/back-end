from flask import Blueprint, request, jsonify
from src.services.user_service import UserService
from src.middlewares.auth_middleware import admin_required
from flask_jwt_extended import jwt_required, get_jwt_identity

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

@user_bp.route('/me/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    from src.models.user_model import User
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    if not user:
        return jsonify({'success': False, 'message': 'User not found'}), 404
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'message': 'No data provided'}), 400
    updated = False
    if 'username' in data:
        user.username = data['username']
        updated = True
    if 'email' in data:
        user.email = data['email']
        updated = True
    if not updated:
        return jsonify({'success': False, 'message': 'No valid fields to update'}), 400
    try:
        from src import db
        db.session.commit()
        return jsonify({'success': True, 'user': user.to_dict(), 'message': 'Profile updated successfully'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 400

@user_bp.route('/me/reset-password', methods=['POST'])
@jwt_required()
def reset_password():
    from src.models.user_model import User
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    if not user:
        return jsonify({'success': False, 'message': 'User not found'}), 404
    data = request.get_json()
    if not data or 'old_password' not in data or 'new_password' not in data:
        return jsonify({'success': False, 'message': 'Old and new password are required'}), 400
    if not user.check_password(data['old_password']):
        return jsonify({'success': False, 'message': 'Old password is incorrect'}), 400
    user.set_password(data['new_password'])
    try:
        from src import db
        db.session.commit()
        return jsonify({'success': True, 'message': 'Password updated successfully'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 400 