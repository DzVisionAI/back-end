from flask import Blueprint, request, jsonify
from src.services.video_service import VideoService
from werkzeug.utils import secure_filename

video_bp = Blueprint('video', __name__)

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
        video_service = VideoService()
        result = video_service.process_video(video_file)
        return jsonify(result), 200 if result['success'] else 400
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500 