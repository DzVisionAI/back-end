from src import db

class Camera(db.Model):
    __tablename__ = 'cameras'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    type = db.Column(db.Integer)
    ipAddress = db.Column(db.String(50))
    macAddress = db.Column(db.String(50))
    status = db.Column(db.String(50))
    location = db.Column(db.String(255))
    typeName = db.Column(db.String(100))
    description = db.Column(db.String(255))
    
    # Relationships
    permissions = db.relationship('UsersPermissions', backref='camera', lazy=True)
    events = db.relationship('Event', backref='camera', lazy=True)
    license_plates = db.relationship('LicensePlate', backref='camera', lazy=True)
    face_recognitions = db.relationship('FaceRecognition', backref='camera', lazy=True)

    def __repr__(self):
        return f'<Camera id={self.id} name={self.name}>' 