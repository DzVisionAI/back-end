from src.models.blacklist_model import BlackList, db
from src.models.user_model import User

class BlackListService:
    @staticmethod
    def get_all():
        try:
            blacklists = BlackList.query.all()
            data = [
                {
                    'id': b.id,
                    'plateNumber': b.plateNumber,
                    'reason': b.reason,
                    'createAt': b.createAt,
                    'status': b.status,
                    'addedBy': {
                        'id': b.addedBy,
                        'username': b.added_by_user.username if b.added_by_user else None
                    }
                } for b in blacklists
            ]
            return {'success': True, 'data': data}
        except Exception as e:
            return {'success': False, 'message': str(e)}

    @staticmethod
    def get_by_id(blacklist_id):
        b = BlackList.query.get(blacklist_id)
        if not b:
            return {'success': False, 'message': 'Not found'}
        data = {
            'id': b.id,
            'plateNumber': b.plateNumber,
            'reason': b.reason,
            'createAt': b.createAt,
            'status': b.status,
            'addedBy': {
                'id': b.addedBy,
                'username': b.added_by_user.username if b.added_by_user else None
            }
        }
        return {'success': True, 'data': data}

    @staticmethod
    def create(data):
        try:
            b = BlackList(
                plateNumber=data['plateNumber'],
                reason=data.get('reason'),
                status=data.get('status'),
                addedBy=data['addedBy']
            )
            db.session.add(b)
            db.session.commit()
            return {'success': True, 'message': 'BlackList entry created', 'data': {'id': b.id}}
        except Exception as e:
            db.session.rollback()
            return {'success': False, 'message': str(e)}

    @staticmethod
    def update(blacklist_id, data):
        b = BlackList.query.get(blacklist_id)
        if not b:
            return {'success': False, 'message': 'Not found'}
        try:
            if 'plateNumber' in data:
                b.plateNumber = data['plateNumber']
            if 'reason' in data:
                b.reason = data['reason']
            if 'status' in data:
                b.status = data['status']
            if 'addedBy' in data:
                b.addedBy = data['addedBy']
            db.session.commit()
            return {'success': True, 'message': 'BlackList entry updated'}
        except Exception as e:
            db.session.rollback()
            return {'success': False, 'message': str(e)}

    @staticmethod
    def delete(blacklist_id, user_id):
        b = BlackList.query.get(blacklist_id)
        if not b:
            return {'success': False, 'message': 'Not found'}
        try:
            db.session.delete(b)
            db.session.commit()
            return {'success': True, 'message': 'BlackList entry deleted'}
        except Exception as e:
            db.session.rollback()
            return {'success': False, 'message': str(e)} 