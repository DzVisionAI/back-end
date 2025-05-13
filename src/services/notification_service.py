from src.models.notification_model import Notification, db
from src.models.user_model import User
from datetime import datetime

class NotificationService:
    @staticmethod
    def get_for_user(user_id):
        notifications = Notification.query.filter_by(userId=user_id).order_by(Notification.sentTime.desc()).all()
        data = [
            {
                'id': n.id,
                'alertId': n.alertId,
                'sentTime': n.sentTime,
                'read': n.read
            } for n in notifications
        ]
        return {'success': True, 'data': data}

    @staticmethod
    def mark_as_read(notification_id, user_id):
        n = Notification.query.filter_by(id=notification_id, userId=user_id).first()
        if not n:
            return {'success': False, 'message': 'Notification not found'}
        n.read = True
        db.session.commit()
        return {'success': True, 'message': 'Notification marked as read'}

    @staticmethod
    def create_for_all_users(alert_id):
        users = User.query.all()
        now = datetime.utcnow()
        for user in users:
            n = Notification(userId=user.id, alertId=alert_id, sentTime=now)
            db.session.add(n)
        db.session.commit()
        return {'success': True, 'message': 'Notifications sent to all users'} 