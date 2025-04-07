from datetime import datetime
from src import db

class FaceRecognition(db.Model):
    __tablename__ = 'face_recognition'
    
    id = db.Column(db.Integer, primary_key=True)
    cameraId = db.Column(db.Integer, db.ForeignKey('cameras.id'))
    driverId = db.Column(db.Integer, db.ForeignKey('drivers.id'))
    detectedAt = db.Column(db.DateTime, default=datetime.utcnow)
    score = db.Column(db.Float)

    def __repr__(self):
        return f'<FaceRecognition id={self.id} driver_id={self.driverId}>' 