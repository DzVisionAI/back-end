from flask import Blueprint, request, jsonify
from flask_jwt_extended import verify_jwt_in_request
from src.services.alert_service import AlertService

alert_bp = Blueprint('alert', __name__)

@alert_bp.route('/', methods=['GET'])
def get_alerts():
    verify_jwt_in_request()
    result = AlertService.get_all()
    return jsonify(result), 200 if result['success'] else 400

@alert_bp.route('/<int:alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id):
    verify_jwt_in_request()
    result = AlertService.mark_as_acknowledged(alert_id)
    return jsonify(result), 200 if result['success'] else 404 