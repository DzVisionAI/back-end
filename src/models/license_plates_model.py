from datetime import datetime
from src import db

class LicensePlate(db.Model):
    __tablename__ = 'license_plates'
    
    id = db.Column(db.Integer, primary_key=True)
    cameraId = db.Column(db.Integer, db.ForeignKey('cameras.id'))
    plateNumber = db.Column(db.String(20), unique=True)
    detectedAt = db.Column(db.DateTime, default=datetime.utcnow)
    image = db.Column(db.String(255))  # Path to the image

    # One-to-one relationship with Vehicle
    vehicle = db.relationship('Vehicle', back_populates='license_plate', uselist=False)

    def __repr__(self):
        return f'<LicensePlate plate={self.plateNumber}>' 