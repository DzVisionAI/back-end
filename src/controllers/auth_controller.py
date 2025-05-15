from flask import Blueprint, request, jsonify
from src.services.auth_service import AuthService
from flask_jwt_extended import jwt_required

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data or 'email' not in data or 'password' not in data:
        return jsonify({
            'success': False,
            'message': 'Email and password are required'
        }), 400

    result = AuthService.login(data['email'], data['password'])
    if result['success']:
        return jsonify(result), 200
    return jsonify(result), 401

@auth_bp.route('/forgot-password', methods=['POST'])
def forgot_password():
    data = request.get_json()
    if not data or 'email' not in data:
        return jsonify({
            'success': False,
            'message': 'Email is required'
        }), 400

    result = AuthService.request_password_reset(data['email'])
    return jsonify(result), 200 if result['success'] else 404

@auth_bp.route('/reset-password', methods=['POST'])
def reset_password():
    data = request.get_json()
    if not data or 'token' not in data or 'new_password' not in data:
        return jsonify({
            'success': False,
            'message': 'Token and new password are required'
        }), 400

    result = AuthService.reset_password(data['token'], data['new_password'])
    return jsonify(result), 200 if result['success'] else 400

@auth_bp.route('/validate-reset-token', methods=['POST'])
def validate_reset_token():
    data = request.get_json()
    if not data or 'token' not in data:
        return jsonify({
            'success': False,
            'message': 'Token is required'
        }), 400

    result = AuthService.validate_reset_token(data['token'])
    return jsonify(result), 200 if result['success'] else 400

@auth_bp.route('/me', methods=['GET'])
@jwt_required()
def get_current_user():
    from flask_jwt_extended import get_jwt_identity
    from src.models.user_model import User
    
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    
    if not user:
        return jsonify({
            'success': False,
            'message': 'User not found'
        }), 404
        
    return jsonify({
        'success': True,
        'user': {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'role': user.role.name
        }
    }), 200
