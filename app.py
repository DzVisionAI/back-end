from src import config, app, socketio

if __name__ == "__main__":
    socketio.run(app, host= config.HOST,
            port= config.PORT,
            debug=config.DEBUG)
    debug_mode = str(config.DEBUG).lower() == 'true'
    app.run(host= config.HOST,
            port= config.PORT,
            debug=debug_mode)