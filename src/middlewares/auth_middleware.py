from functools import wraps
from flask import jsonify
from flask_jwt_extended import get_jwt_identity, verify_jwt_in_request
from src.models.user_model import User, UserRole

# Authentication and admin middleware for route protection
def admin_required():
    def wrapper(fn):
        @wraps(fn)
        def decorator(*args, **kwargs):
            verify_jwt_in_request()
            user_id = get_jwt_identity()
            user = User.query.get(user_id)
            
            if not user or not user.is_admin():
                return jsonify({
                    'success': False,
                    'message': 'Admin access required'
                }), 403
            
            return fn(*args, **kwargs)
        return decorator
    return wrapper 