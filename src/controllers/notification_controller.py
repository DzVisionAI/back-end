from flask import Blueprint, request, jsonify
from flask_jwt_extended import get_jwt_identity, verify_jwt_in_request
from src.services.notification_service import NotificationService

notification_bp = Blueprint('notification', __name__)

@notification_bp.route('/', methods=['GET'])
def get_notifications():
    verify_jwt_in_request()
    user_id = get_jwt_identity()
    result = NotificationService.get_for_user(user_id)
    return jsonify(result), 200 if result['success'] else 400

@notification_bp.route('/<int:notification_id>/read', methods=['POST'])
def mark_notification_as_read(notification_id):
    verify_jwt_in_request()
    user_id = get_jwt_identity()
    result = NotificationService.mark_as_read(notification_id, user_id)
    return jsonify(result), 200 if result['success'] else 404 