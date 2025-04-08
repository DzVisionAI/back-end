from src import config, app

if __name__ == "__main__":
    debug_mode = str(config.DEBUG).lower() == 'true'
    app.run(host= config.HOST,
            port= config.PORT,
            debug=debug_mode)