from datetime import datetime
from src import db
from src.models.user_model import User
from flask_mail import Message
from src import mail
from flask import current_app

class AuthService:
    @staticmethod
    def login(email, password):
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            return {
                'success': True,
                'token': user.get_token(),
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'role': user.role.value
                }
            }
        return {
            'success': False,
            'message': 'Invalid email or password'
        }

    @staticmethod
    def request_password_reset(email):
        user = User.query.filter_by(email=email).first()
        if not user:
            return {
                'success': False,
                'message': 'Email not found'
            }

        token = user.generate_reset_token()
        db.session.commit()

        # Send reset email
        reset_url = f"{current_app.config['FRONTEND_URL']}/reset-password?token={token}"
        msg = Message(
            'Password Reset Request',
            sender=current_app.config['MAIL_DEFAULT_SENDER'],
            recipients=[user.email]
        )
        msg.body = f'''To reset your password, visit the following link:
{reset_url}

If you did not make this request, please ignore this email.
'''
        mail.send(msg)

        return {
            'success': True,
            'message': 'Password reset instructions sent to your email'
        }

    @staticmethod
    def reset_password(token, new_password):
        user = User.query.filter_by(reset_token=token).first()
        if not user or user.reset_token_expires < datetime.utcnow():
            return {
                'success': False,
                'message': 'Invalid or expired reset token'
            }

        user.set_password(new_password)
        user.reset_token = None
        user.reset_token_expires = None
        db.session.commit()

        return {
            'success': True,
            'message': 'Password has been reset successfully'
        }

    @staticmethod
    def validate_reset_token(token):
        user = User.query.filter_by(reset_token=token).first()
        if not user or user.reset_token_expires < datetime.utcnow():
            return {
                'success': False,
                'message': 'Invalid or expired reset token'
            }
        return {
            'success': True,
            'message': 'Valid reset token'
        } 