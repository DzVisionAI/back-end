import os

class ProductionConfig:
    def __init__(self):
        self.ENV = os.environ.get("FLASK_ENV", "production")
        self.DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() == "True"
        self.PORT = int(os.environ.get("PORT", 80))
        self.HOST = os.environ.get("HOST", "0.0.0.0")