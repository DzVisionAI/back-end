from src.models.alerts_model import Alert, db
from datetime import datetime

class AlertService:
    @staticmethod
    def get_all():
        alerts = Alert.query.order_by(Alert.time.desc()).all()
        data = [
            {
                'id': a.id,
                'eventId': a.eventId,
                'time': a.time,
                'status': a.status,
                'acknowledged': a.acknowledged
            } for a in alerts
        ]
        return {'success': True, 'data': data}

    @staticmethod
    def mark_as_acknowledged(alert_id):
        a = Alert.query.get(alert_id)
        if not a:
            return {'success': False, 'message': 'Alert not found'}
        a.acknowledged = True
        db.session.commit()
        return {'success': True, 'message': 'Alert acknowledged'}

    @staticmethod
    def create(event_id, status):
        a = Alert(eventId=event_id, status=status, time=datetime.utcnow())
        db.session.add(a)
        db.session.commit()
        return {'success': True, 'message': 'Alert created', 'data': {'id': a.id}} 