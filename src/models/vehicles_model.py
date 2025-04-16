from datetime import datetime
from src import db

class Vehicle(db.Model):
    __tablename__ = 'vehicles'
    
    id = db.Column(db.Integer, primary_key=True)
    plateNumber = db.Column(db.String(20))
    color = db.Column(db.String(50))
    make = db.Column(db.String(100))
    model = db.Column(db.String(100))
    ownerId = db.Column(db.Integer)
    registerAt = db.Column(db.DateTime, default=datetime.utcnow)
    image = db.Column(db.String(500))  # Path to vehicle image
    
    # Relationships
    drivers = db.relationship('Driver', backref='vehicle', lazy=True)

    def __repr__(self):
        return f'<Vehicle plate={self.plateNumber}, make={self.make}, color={self.color}>' 