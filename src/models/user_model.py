from datetime import datetime
from src import db

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.Integer, nullable=False)
    createdAt = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    permissions = db.relationship('UsersPermissions', backref='user', lazy=True)
    notifications = db.relationship('Notification', backref='user', lazy=True)
    blacklists = db.relationship('BlackList', backref='added_by_user', lazy=True)

    def __repr__(self):
        return f'<User {self.username}>'