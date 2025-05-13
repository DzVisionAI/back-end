from datetime import datetime
from src import db
from marshmallow import Schema, fields, validates, ValidationError
import re

# BlackList model for storing blacklisted vehicle plates
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

class BlackListSchema(Schema):
    id = fields.Int(dump_only=True)
    plateNumber = fields.Str(required=True)
    reason = fields.Str()
    createAt = fields.DateTime(dump_only=True)
    status = fields.Str()
    addedBy = fields.Int(dump_only=True)

    @validates('plateNumber')
    def validate_plate_number(self, value):
        if not re.match(r'^[A-Za-z0-9]+$', value):
            raise ValidationError('Plate number must be alphanumeric.') 