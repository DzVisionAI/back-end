from datetime import datetime
from src import db

class Alert(db.Model):
    __tablename__ = 'alerts'
    
    id = db.Column(db.Integer, primary_key=True)
    eventId = db.Column(db.Integer, db.ForeignKey('events.id'))
    time = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(50))
    
    # Relationships
    notifications = db.relationship('Notification', backref='alert', lazy=True)

    def __repr__(self):
        return f'<Alert id={self.id} event_id={self.eventId}>' 