from datetime import datetime
from src import db

class Driver(db.Model):
    __tablename__ = 'drivers'
    
    id = db.Column(db.Integer, primary_key=True)
    cameraId = db.Column(db.Integer, db.ForeignKey('cameras.id'))
    vehicleId = db.Column(db.Integer, db.ForeignKey('vehicles.id'))
    behaviourType = db.Column(db.String(50))
    image = db.Column(db.String(255))  # Path to the image
    detectedAt = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    events = db.relationship('Event', backref='driver', lazy=True)
    face_recognitions = db.relationship('FaceRecognition', backref='driver', lazy=True)

    def __repr__(self):
        return f'<Driver id={self.id}>' 