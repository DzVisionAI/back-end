from datetime import datetime
from src import db
from marshmallow import Schema, fields

class Notification(db.Model):
    __tablename__ = 'notification'
    
    id = db.Column(db.Integer, primary_key=True)
    userId = db.Column(db.Integer, db.ForeignKey('users.id'))
    alertId = db.Column(db.Integer, db.ForeignKey('alerts.id'))
    sentTime = db.Column(db.DateTime, default=datetime.utcnow)
    read = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f'<Notification id={self.id} user_id={self.userId}>'

class NotificationSchema(Schema):
    id = fields.Int(dump_only=True)
    userId = fields.Int(required=True)
    alertId = fields.Int(required=True)
    sentTime = fields.DateTime(dump_only=True)
    read = fields.Bool() 