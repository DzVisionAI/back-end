from src import db

class UsersPermissions(db.Model):
    __tablename__ = 'users_permissions'
    
    id = db.Column(db.Integer, primary_key=True)
    userId = db.Column(db.Integer, db.ForeignKey('users.id'))
    canView = db.Column(db.Boolean, default=False)
    canControl = db.Column(db.Boolean, default=False)
    cameraId = db.Column(db.Integer, db.ForeignKey('cameras.id'))

    def __repr__(self):
        return f'<UsersPermissions user_id={self.userId} camera_id={self.cameraId}>' 