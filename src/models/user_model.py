from datetime import datetime, timedelta
from src import db
import bcrypt
from flask_jwt_extended import create_access_token
import secrets
from enum import IntEnum

class UserRole(IntEnum):
    USER = 0
    ADMIN = 1

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.Integer, nullable=False, default=UserRole.USER)
    createdAt = db.Column(db.DateTime, default=datetime.utcnow)
    reset_token = db.Column(db.String(100), unique=True)
    reset_token_expires = db.Column(db.DateTime)
    
    # Relationships
    permissions = db.relationship('UsersPermissions', backref='user', lazy=True)
    notifications = db.relationship('Notification', backref='user', lazy=True)
    blacklists = db.relationship('BlackList', backref='added_by_user', lazy=True)

    def set_password(self, password):
        salt = bcrypt.gensalt()
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))

    def get_token(self):
        return create_access_token(identity=self.id)

    def generate_reset_token(self):
        # Generate a random token
        token = secrets.token_urlsafe(32)
        # Set token and expiration (24 hours from now)
        self.reset_token = token
        self.reset_token_expires = datetime.utcnow() + timedelta(hours=24)
        return token

    def is_admin(self):
        return self.role == UserRole.ADMIN

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'createdAt': self.createdAt.isoformat()
        }

    def __repr__(self):
        return f'<User {self.username}>'