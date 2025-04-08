import os

class DevConfig:
    def __init__(self):
        self.ENV = os.environ.get("FLASK_ENV", "development")
        self.DEBUG = os.environ.get("FLASK_DEBUG", "True").lower() == "true"
        self.PORT = int(os.environ.get("PORT", 3000))
        self.HOST = os.environ.get("HOST", "0.0.0.0")