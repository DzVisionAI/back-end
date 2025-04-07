from datetime import datetime
from src import db

class Notification(db.Model):
    __tablename__ = 'notification'
    
    id = db.Column(db.Integer, primary_key=True)
    userId = db.Column(db.Integer, db.ForeignKey('users.id'))
    alertId = db.Column(db.Integer, db.ForeignKey('alerts.id'))
    sentTime = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Notification id={self.id} user_id={self.userId}>' 