from flask import Blueprint, request, jsonify, current_app
from src.services.video_service import VideoService
from werkzeug.utils import secure_filename
import os
from src.models.vehicles_model import Vehicle
from src.models.license_plates_model import LicensePlate
from flask_socketio import emit
from threading import Thread
from flask import copy_current_request_context
from src import socketio
from src import app
import datetime
import json
video_bp = Blueprint('video', __name__)

# try:
#     from src import socketio
# except ImportError:
#     socketio = None  # fallback for testing, but should be imported in real app

@video_bp.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({
            'success': False,
            'message': 'No video file provided'
        }), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({
            'success': False,
            'message': 'No selected file'
        }), 400

    # Check file extension
    allowed_extensions = {'mp4', 'avi', 'mov'}
    if '.' not in video_file.filename or \
       video_file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({
            'success': False,
            'message': 'Invalid file type. Allowed types: mp4, avi, mov'
        }), 400

    try:
        # Secure the filename and save the file
        filename = secure_filename(video_file.filename)
        upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        video_file.save(upload_path)
        
        return jsonify({
            'success': True,
            'message': 'Video uploaded successfully',
            'filename': filename
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@video_bp.route('/process/<filename>', methods=['POST'])
def process_video(filename):
    try:
        # Check if file exists
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'message': 'Video file not found'
            }), 404

        video_service = VideoService()
        result = video_service.process_video(file_path)

        # If processing failed, return error
        if not result.get('success', False):
            return jsonify(result), 400

        # Return the detection results directly from the process_video output
        return jsonify({
            'success': True,
            'message': 'Video processed and results saved',
            'detections': result.get('detections', []),
            'frames_processed': result.get('frames_processed'),
            'vehicles_detected': result.get('vehicles_detected'),
            'plates_detected': result.get('plates_detected'),
            'average_fps': result.get('average_fps'),
            'processing_time_per_frame': result.get('processing_time_per_frame'),
            'total_processing_time': result.get('total_processing_time')
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500 

@socketio.on('start_processing', namespace='/video')
def handle_start_processing(data):
    print("Starting processing")
    try:
        filename = data.get('filename')
        if not filename:
            print("Error: Filename not provided")
            socketio.emit('error', {'message': 'Filename not provided'}, namespace='/video')
            return

        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            print(f"Error: Video file not found at {file_path}")
            socketio.emit('error', {'message': 'Video file not found'}, namespace='/video')
            return

        video_service = VideoService()

        def process_and_emit():
            with app.app_context():
                try:
                    for event in video_service.process_video_realtime(file_path):
                        # Convert datetime objects to ISO strings recursively
                        # def convert_dt(obj):
                        #     if isinstance(obj, dict):
                        #         return {k: convert_dt(v) for k, v in obj.items()}
                        #     elif isinstance(obj, list):
                        #         return [convert_dt(i) for i in obj]
                        #     elif isinstance(obj, datetime.datetime):
                        #         return obj.isoformat()
                        #     else:
                        #         return obj
                        # event_serializable = convert_dt(event)
                        event_serializable = json.dumps(event, default=str)
                        print("Emitting event:", event_serializable)  # DEBUG
                        socketio.emit('video_event', event_serializable, namespace='/video')
                    print("Emitting processing_complete")  # DEBUG
                    socketio.emit('processing_complete', namespace='/video')
                except Exception as e:
                    print("Error in process_and_emit:", e)  # DEBUG
                    socketio.emit('error', {'message': str(e)}, namespace='/video')

        # Start processing in a background thread
        thread = Thread(target=process_and_emit)
        thread.start()

        print("Emitting processing_started")  # DEBUG
        socketio.emit('processing_started', namespace='/video')
    except Exception as e:
        print("Error in handle_start_processing:", e)  # DEBUG
        socketio.emit('error', {'message': str(e)}, namespace='/video')
        
        


@socketio.on('message', namespace='/video')
def handle_message(message):
    try:
        print(f"Received message: {message}")  # DEBUG
        # Echo the message back to client
        socketio.emit('message', "hello", namespace='/video')
    except Exception as e:
        print(f"Error in handle_message: {e}")  # DEBUG
        socketio.emit('error', {'message': str(e)}, namespace='/video')

@video_bp.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'message': 'No image file provided'
        }), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({
            'success': False,
            'message': 'No selected file'
        }), 400

    allowed_extensions = {'jpg', 'jpeg', 'png', 'bmp'}
    if '.' not in image_file.filename or \
       image_file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({
            'success': False,
            'message': 'Invalid file type. Allowed types: jpg, jpeg, png, bmp'
        }), 400

    try:
        filename = secure_filename(image_file.filename)
        upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        image_file.save(upload_path)
        return jsonify({
            'success': True,
            'message': 'Image uploaded successfully',
            'filename': filename
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@video_bp.route('/process_image/<filename>', methods=['POST'])
def process_image(filename):
    try:
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'message': 'Image file not found'
            }), 404

        video_service = VideoService()
        result = video_service.process_image(file_path)

        if not result.get('success', False):
            return jsonify(result), 400

        return jsonify({
            'success': True,
            'message': 'Image processed and results saved',
            'detections': result.get('detections', []),
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500
