from src import app, db
from src.models.user_model import User, UserRole

def seed_users():
    # Create initial users
    users = [
                User(username='lokmane', email='l.zeddoun@esi-sba.dz', role=UserRole.ADMIN)
    ]

    # Set passwords for users
    users[0].set_password('pass123')

    # Use application context to interact with the database
    with app.app_context():
        # Add users to the session and commit
        with db.session.begin():
            db.session.add_all(users)

    print("Users seeded successfully.")

if __name__ == "__main__":
    seed_users() 