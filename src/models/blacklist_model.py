from datetime import datetime
from src import db

class BlackList(db.Model):
    __tablename__ = 'blacklist'
    
    id = db.Column(db.Integer, primary_key=True)
    plateNumber = db.Column(db.String(255), nullable=False)
    reason = db.Column(db.String(255))
    createAt = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(50))
    addedBy = db.Column(db.Integer, db.ForeignKey('users.id'))

    def __repr__(self):
        return f'<BlackList {self.plateNumber}>' 