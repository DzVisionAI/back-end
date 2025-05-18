from flask import Flask
import os
from src.config.config import Config
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager
from flask_mail import Mail
from flask_cors import CORS
from datetime import timedelta

load_dotenv()

# declaring flask application
app = Flask(__name__)

# Enable CORS
CORS(app)

# calling the dev configuration
config = Config().dev_config

# making our application to use dev env
app.env = config.ENV

# Path for our local sql lite database
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("SQLALCHEMY_DATABASE_URI_DEV")

# To specify to track modifications of objects and emit signals
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = os.environ.get("SQLALCHEMY_TRACK_MODIFICATIONS")

# JWT Configuration
app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "your-secret-key")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=24)

# Mail Configuration
app.config["MAIL_SERVER"] = os.environ.get("MAIL_SERVER", "smtp.gmail.com")
app.config["MAIL_PORT"] = int(os.environ.get("MAIL_PORT", 587))
app.config["MAIL_USE_TLS"] = os.environ.get("MAIL_USE_TLS", True)
app.config["MAIL_USERNAME"] = os.environ.get("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.environ.get("MAIL_PASSWORD")
app.config["MAIL_DEFAULT_SENDER"] = os.environ.get("MAIL_DEFAULT_SENDER")

# Frontend URL for password reset
app.config["FRONTEND_URL"] = os.environ.get("FRONTEND_URL", "http://localhost:3000")

# Video upload configuration
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max file size
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uploads")

# Create uploads directory if it doesn't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# sql alchemy instance
db = SQLAlchemy(app)

# Flask Migrate instance to handle migrations
migrate = Migrate(app, db)

# Initialize extensions
jwt = JWTManager(app)
mail = Mail(app)

# import models to let the migrate tool know
from src.models import (
    User, BlackList, UsersPermissions, Notification, Alert,
    Event, LicensePlate, Camera, FaceRecognition, Driver, Vehicle
)

# Register blueprints
from src.controllers.auth_controller import auth_bp
from src.controllers.user_controller import user_bp
from src.controllers.video_controller import video_bp
from src.controllers.blacklist_controller import blacklist_bp
from src.controllers.notification_controller import notification_bp
from src.controllers.alert_controller import alert_bp
from src.controllers.vehicles_controller import vehicles_bp
from src.controllers.license_plates_controller import license_plates_bp
from src.controllers.events_controller import events_bp
from src.controllers.drivers_controller import drivers_bp
from src.controllers.cameras_controller import cameras_bp

app.register_blueprint(auth_bp, url_prefix='/api/auth')
app.register_blueprint(user_bp, url_prefix='/api/users')
app.register_blueprint(video_bp, url_prefix='/api/video')
app.register_blueprint(blacklist_bp, url_prefix='/api/blacklist')
app.register_blueprint(notification_bp, url_prefix='/api/notification')
app.register_blueprint(alert_bp, url_prefix='/api/alert')
app.register_blueprint(vehicles_bp, url_prefix='/api/vehicles')
app.register_blueprint(license_plates_bp, url_prefix='/api/license-plates')
app.register_blueprint(events_bp, url_prefix='/api/events')
app.register_blueprint(drivers_bp, url_prefix='/api/drivers')
app.register_blueprint(cameras_bp, url_prefix='/api/cameras')