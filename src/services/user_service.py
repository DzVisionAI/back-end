from src import db
from src.models.user_model import User, UserRole
from sqlalchemy.exc import IntegrityError

class UserService:
    @staticmethod
    def get_all_users(page=1, per_page=10):
        users = User.query.paginate(page=page, per_page=per_page)
        return {
            'success': True,
            'users': [user.to_dict() for user in users.items],
            'total': users.total,
            'pages': users.pages,
            'current_page': users.page
        }

    @staticmethod
    def get_user(user_id):
        user = User.query.get(user_id)
        if not user:
            return {
                'success': False,
                'message': 'User not found'
            }
        return {
            'success': True,
            'user': user.to_dict()
        }

    @staticmethod
    def create_user(data):
        try:
            role_value = data.get('role', UserRole.USER)
            # Accept string (e.g., 'admin'), UserRole, or default
            if isinstance(role_value, str):
                role_enum = UserRole(role_value.lower())
            else:
                role_enum = role_value
            user = User(
                username=data['username'],
                email=data['email'],
                role=role_enum
            )
            user.set_password(data['password'])
            
            db.session.add(user)
            db.session.commit()
            
            return {
                'success': True,
                'user': user.to_dict(),
                'message': 'User created successfully'
            }
        except IntegrityError:
            db.session.rollback()
            return {
                'success': False,
                'message': 'Username or email already exists'
            }
        except Exception as e:
            db.session.rollback()
            return {
                'success': False,
                'message': str(e)
            }

    @staticmethod
    def update_user(user_id, data):
        user = User.query.get(user_id)
        if not user:
            return {
                'success': False,
                'message': 'User not found'
            }

        try:
            if 'username' in data:
                user.username = data['username']
            if 'email' in data:
                user.email = data['email']
            if 'password' in data:
                user.set_password(data['password'])
            if 'role' in data:
                role_value = data['role']
                if isinstance(role_value, str):
                    user.role = UserRole(role_value.lower())
                else:
                    user.role = role_value

            db.session.commit()
            return {
                'success': True,
                'user': user.to_dict(),
                'message': 'User updated successfully'
            }
        except IntegrityError:
            db.session.rollback()
            return {
                'success': False,
                'message': 'Username or email already exists'
            }
        except Exception as e:
            db.session.rollback()
            return {
                'success': False,
                'message': str(e)
            }

    @staticmethod
    def delete_user(user_id):
        user = User.query.get(user_id)
        if not user:
            return {
                'success': False,
                'message': 'User not found'
            }
        try:
            db.session.delete(user)
            db.session.commit()
            return {
                'success': True,
                'message': 'User deleted successfully'
            }
        except Exception as e:
            db.session.rollback()
            return {
                'success': False,
                'message': str(e)
            } 