from datetime import datetime
from src import db

class Event(db.Model):
    __tablename__ = 'events'
    
    id = db.Column(db.Integer, primary_key=True)
    cameraId = db.Column(db.Integer, db.ForeignKey('cameras.id'))
    plateId = db.Column(db.Integer)
    time = db.Column(db.DateTime, default=datetime.utcnow)
    typeName = db.Column(db.String(100))
    description = db.Column(db.String(255))
    driverId = db.Column(db.Integer, db.ForeignKey('drivers.id'))
    
    # Relationships
    alerts = db.relationship('Alert', backref='event', lazy=True)

    def __repr__(self):
        return f'<Event id={self.id} type={self.typeName}>' 