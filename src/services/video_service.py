import cv2
import numpy as np
from datetime import datetime
import os
import uuid
from werkzeug.utils import secure_filename
import time

from src import db
from src.models.license_plates_model import LicensePlate
from src.models.events_model import Event
from src.ai import LicensePlateDetector, VehicleTracker, get_car, read_license_plate
from src.models.vehicles_model import Vehicle

# Import new AI functions (you'll need to implement these)
from src.ai import detect_vehicle_color
from src.utils import upload_to_gcs

class VideoService:
    def __init__(self):
        try:
            # Create necessary directories
            self.upload_folder = os.path.join(os.getcwd(), 'uploads')
            self.video_folder = os.path.join(self.upload_folder, 'videos')
            self.plates_folder = os.path.join(self.upload_folder, 'plates')
            
            os.makedirs(self.video_folder, exist_ok=True)
            os.makedirs(self.plates_folder, exist_ok=True)
            
            # Initialize AI components
            try:
                self.detector = LicensePlateDetector()
                self.tracker = VehicleTracker()
                print("All AI components loaded successfully")
            except Exception as model_error:
                print(f"Error loading AI components: {str(model_error)}")
                raise Exception(f"Failed to load AI components: {str(model_error)}")
            
            print("VideoService initialized successfully")
            
        except Exception as e:
            print(f"Error initializing VideoService: {str(e)}")
            raise Exception(f"Failed to initialize video service: {str(e)}")

    def process_video(self, video_path, target_fps=15):
        try:
            # Check if video exists
            if not os.path.exists(video_path):
                return {
                    'success': False,
                    'message': f'Video file not found: {video_path}'
                }

            print(f"Processing video: {video_path}")
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {
                    'success': False,
                    'message': 'Failed to open video file'
                }
            
            # Get video properties
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame interval to achieve target FPS
            target_frame_time = 1.0 / target_fps
            
            # Initialize counters
            frame_number = 0
            vehicles_detected = 0
            plates_detected = 0
            processing_times = []
            last_frame_time = time.time()
            detection_results = []  # Collect detection results
            
            print(f"Original video FPS: {original_fps}, Target FPS: {target_fps}")
            
            # Process video
            while True:
                # Calculate time since last frame
                current_time = time.time()
                elapsed_time = current_time - last_frame_time
                
                # If we're processing faster than target FPS, wait
                if elapsed_time < target_frame_time:
                    time.sleep(target_frame_time - elapsed_time)
                
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_number += 1
                
                try:
                    # Detect vehicles
                    detections = self.detector.detect_vehicles(frame)
                    vehicles_detected += len(detections)
                    
                    # Track vehicles
                    track_ids = self.tracker.update(detections)
                    
                    # Detect license plates
                    license_plates = self.detector.detect_license_plates(frame)
                    
                    for license_plate in license_plates:
                        x1, y1, x2, y2, score, class_id = license_plate
                        
                        # Assign license plate to car
                        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
                        
                        if car_id != -1:
                            try:
                                # Crop license plate
                                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                                if license_plate_crop.size > 0:
                                    # Process license plate
                                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                                    
                                    # Read license plate text
                                    plate_text, text_score = read_license_plate(license_plate_crop_thresh)
                                    
                                    if plate_text is not None:
                                        plates_detected += 1
                                        # Save detection
                                        detection_id = str(uuid.uuid4())
                                        detection_result = self.save_detection(
                                            detection_id=detection_id,
                                            frame_number=frame_number,
                                            vehicle_crop=frame[int(ycar1):int(ycar2), int(xcar1):int(xcar2)],
                                            plate_crop=license_plate_crop,
                                            plate_text=plate_text,
                                            text_score=text_score
                                        )
                                        if detection_result:
                                            detection_results.append(detection_result)
                            except Exception as plate_error:
                                print(f"Error processing plate in frame {frame_number}: {str(plate_error)}")
                                continue
                    
                    # Calculate processing time for this frame
                    end_time = time.time()
                    processing_time = end_time - start_time
                    processing_times.append(processing_time)
                    
                    # Update last frame time
                    last_frame_time = time.time()
                    
                    # Print progress every 30 frames
                    if frame_number % 30 == 0:
                        current_fps = 1.0 / (sum(processing_times[-30:]) / len(processing_times[-30:]))
                        progress = (frame_number / total_frames) * 100
                        print(f"Progress: {progress:.1f}% - Current FPS: {current_fps:.2f}")
                    
                except Exception as frame_error:
                    print(f"Error processing frame {frame_number}: {str(frame_error)}")
                    continue
            
            cap.release()
            
            # Calculate overall statistics
            avg_processing_time = sum(processing_times) / len(processing_times)
            avg_fps = 1.0 / avg_processing_time
            
            return {
                'success': True,
                'message': 'Video processed successfully',
                'frames_processed': frame_number,
                'vehicles_detected': vehicles_detected,
                'plates_detected': plates_detected,
                'average_fps': avg_fps,
                'target_fps': target_fps,
                'original_fps': original_fps,
                'processing_time_per_frame': avg_processing_time,
                'total_processing_time': sum(processing_times),
                'detections': detection_results
            }

        except Exception as e:
            print(f"Error processing video: {str(e)}")
            return {
                'success': False,
                'message': f'Error processing video: {str(e)}'
            }

    def process_video_realtime(self, video_path, target_fps=15):
        """Generator that processes video and yields events for real-time streaming via Socket.IO."""
        try:
            if not os.path.exists(video_path):
                yield {
                    'type': 'error',
                    'data': {'message': f'Video file not found: {video_path}'}
                }
                return

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                yield {
                    'type': 'error',
                    'data': {'message': 'Failed to open video file'}
                }
                return

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            target_frame_time = 1.0 / target_fps
            frame_number = 0
            vehicles_detected = 0
            plates_detected = 0
            processing_times = []
            last_frame_time = time.time()

            while True:
                current_time = time.time()
                elapsed_time = current_time - last_frame_time
                if elapsed_time < target_frame_time:
                    time.sleep(target_frame_time - elapsed_time)

                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                frame_number += 1
                try:
                    detections = self.detector.detect_vehicles(frame)
                    vehicles_detected += len(detections)
                    track_ids = self.tracker.update(detections)
                    license_plates = self.detector.detect_license_plates(frame)

                    for license_plate in license_plates:
                        x1, y1, x2, y2, score, class_id = license_plate
                        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
                        if car_id != -1:
                            try:
                                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                                if license_plate_crop.size > 0:
                                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                                    plate_text, text_score = read_license_plate(license_plate_crop_thresh)
                                    if plate_text is not None:
                                        plates_detected += 1
                                        detection_id = str(uuid.uuid4())
                                        detection_result = self.save_detection(
                                            detection_id=detection_id,
                                            frame_number=frame_number,
                                            vehicle_crop=frame[int(ycar1):int(ycar2), int(xcar1):int(xcar2)],
                                            plate_crop=license_plate_crop,
                                            plate_text=plate_text,
                                            text_score=text_score
                                        )
                                        if detection_result:
                                            yield {
                                                'type': 'detection',
                                                'data': detection_result
                                            }
                            except Exception as plate_error:
                                yield {
                                    'type': 'error',
                                    'data': {'message': f'Error processing plate in frame {frame_number}: {str(plate_error)}'}
                                }
                                continue

                    end_time = time.time()
                    processing_time = end_time - start_time
                    processing_times.append(processing_time)
                    last_frame_time = time.time()

                    if frame_number % 30 == 0:
                        current_fps = 1.0 / (sum(processing_times[-30:]) / len(processing_times[-30:])) if len(processing_times[-30:]) > 0 else 0
                        progress = (frame_number / total_frames) * 100 if total_frames > 0 else 0
                        yield {
                            'type': 'progress',
                            'data': {
                                'frame_number': frame_number,
                                'progress': progress,
                                'current_fps': current_fps
                            }
                        }
                except Exception as frame_error:
                    yield {
                        'type': 'error',
                        'data': {'message': f'Error processing frame {frame_number}: {str(frame_error)}'}
                    }
                    continue

            cap.release()
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
            avg_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
            yield {
                'type': 'complete',
                'data': {
                    'frames_processed': frame_number,
                    'vehicles_detected': vehicles_detected,
                    'plates_detected': plates_detected,
                    'average_fps': avg_fps,
                    'processing_time_per_frame': avg_processing_time,
                    'total_processing_time': sum(processing_times)
                }
            }
        except Exception as e:
            yield {
                'type': 'error',
                'data': {'message': f'Error processing video: {str(e)}'}
            }

    def save_detection(self, detection_id, frame_number, vehicle_crop, plate_crop, plate_text, text_score):
        """Save detection results to files, upload to GCS, and database with proper model relationships."""
        try:
            # Generate unique filenames with timestamp
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            vehicle_filename = f'vehicle_{detection_id}_{timestamp}_{frame_number}.jpg'
            plate_filename = f'plate_{detection_id}_{timestamp}_{frame_number}.jpg'
            
            # Create full paths
            vehicle_path = os.path.join(self.plates_folder, vehicle_filename)
            plate_path = os.path.join(self.plates_folder, plate_filename)
            
            # Save images with error checking
            if not cv2.imwrite(vehicle_path, vehicle_crop):
                raise Exception(f"Failed to save vehicle image to {vehicle_path}")
            if not cv2.imwrite(plate_path, plate_crop):
                raise Exception(f"Failed to save plate image to {plate_path}")
            
            print(f"Saved images - Vehicle: {vehicle_path}, Plate: {plate_path}")
            
            # Upload to GCS
            try:
                vehicle_gcs = upload_to_gcs(vehicle_path, f"vehicles/{vehicle_filename}")
                vehicle_gcs_url = vehicle_gcs["blob_url"]
                vehicle_signed_url = vehicle_gcs["signed_url"]
            except Exception as gcs_e:
                print(f"Error uploading vehicle image to GCS: {gcs_e}")
                vehicle_gcs_url = None
                vehicle_signed_url = None
            try:
                plate_gcs = upload_to_gcs(plate_path, f"plates/{plate_filename}")
                plate_gcs_url = plate_gcs["blob_url"]
                plate_signed_url = plate_gcs["signed_url"]
            except Exception as gcs_e:
                print(f"Error uploading plate image to GCS: {gcs_e}")
                plate_gcs_url = None
                plate_signed_url = None
            
            # Detect vehicle color
            try:
                vehicle_color = detect_vehicle_color(vehicle_crop)
                print(f"Detected vehicle color: {vehicle_color}")
            except Exception as ai_error:
                print(f"Error detecting vehicle color: {str(ai_error)}")
                vehicle_color = None
            
            detection_time = datetime.utcnow()
            
            try:
                # First, check if license plate exists
                license_plate = LicensePlate.query.filter_by(plateNumber=plate_text).first()
                if not license_plate:
                    license_plate = LicensePlate(
                        plateNumber=plate_text,
                        detectedAt=detection_time,
                        image=plate_gcs_url or plate_path,
                        cameraId=None
                    )
                    db.session.add(license_plate)
                    db.session.flush()
                else:
                    # Optionally update detectedAt/image if you want to keep latest
                    license_plate.detectedAt = detection_time
                    license_plate.image = plate_gcs_url or plate_path
                    db.session.flush()

                # Now, check if a vehicle exists with this license_plate_id
                vehicle = Vehicle.query.filter_by(license_plate_id=license_plate.id).first()
                if not vehicle:
                    vehicle = Vehicle(
                        license_plate_id=license_plate.id,
                        registerAt=detection_time,
                        color=vehicle_color,
                        model=None,  # This could be enhanced with model detection
                        ownerId=None,
                        image=vehicle_gcs_url or vehicle_path  # Save the GCS url or fallback to local path
                    )
                    db.session.add(vehicle)
                    db.session.flush()
                    # Create event for new vehicle registration
                    registration_event = Event(
                        typeName='new_vehicle_registration',
                        description=f'New vehicle registered with plate {plate_text}, color: {vehicle_color}',
                        time=detection_time,
                        plateId=license_plate.id,
                        cameraId=None,
                        driverId=None
                    )
                    db.session.add(registration_event)
                else:
                    # Update existing vehicle information if new data is available
                    if vehicle_color and not vehicle.color:
                        vehicle.color = vehicle_color
                        # Create event for color update
                        color_event = Event(
                            typeName='vehicle_color_update',
                            description=f'Vehicle color detected: {vehicle_color}',
                            time=detection_time,
                            plateId=license_plate.id,
                            cameraId=None,
                            driverId=None
                        )
                        db.session.add(color_event)
                    # Optionally update vehicle image
                    vehicle.image = vehicle_gcs_url or vehicle_path
                    db.session.flush()

                # Create detection event
                detection_event = Event(
                    typeName='license_plate_detection',
                    description=f'License plate {plate_text} detected in frame {frame_number}',
                    time=detection_time,
                    plateId=license_plate.id,
                    cameraId=None,
                    driverId=None
                )
                db.session.add(detection_event)
                
                # Commit all changes
                db.session.commit()
                print(f"Successfully saved to database - Vehicle: {vehicle.id}, Plate: {plate_text}")
                
                return {
                    'success': True,
                    'vehicle': {
                        'id': vehicle.id,
                        'plate_number': license_plate.plateNumber,
                        'color': vehicle.color,
                        'image_path': vehicle_gcs_url or vehicle_path,
                        'gcs_url': vehicle_gcs_url,
                        'signed_url': vehicle_signed_url
                    },
                    'license_plate': {
                        'id': license_plate.id,
                        'number': plate_text,
                        'image_path': plate_gcs_url or plate_path,
                        'gcs_url': plate_gcs_url,
                        'signed_url': plate_signed_url
                    },
                    'frame_number': frame_number,
                    'detection_time': detection_time
                }
                
            except Exception as db_error:
                db.session.rollback()
                print(f"Database error: {str(db_error)}")
                # Clean up saved images if database save failed
                try:
                    if os.path.exists(vehicle_path):
                        os.remove(vehicle_path)
                    if os.path.exists(plate_path):
                        os.remove(plate_path)
                except Exception as cleanup_error:
                    print(f"Error cleaning up images after failed database save: {str(cleanup_error)}")
                raise Exception(f"Failed to save to database: {str(db_error)}")
            
        except Exception as e:
            print(f"Error in save_detection: {str(e)}")
            db.session.rollback()
            raise e

    def process_image(self, image_path):
        try:
            if not os.path.exists(image_path):
                return {
                    'success': False,
                    'message': f'Image file not found: {image_path}'
                }
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'success': False,
                    'message': 'Failed to load image file'
                }
            detections = self.detector.detect_vehicles(image)
            track_ids = self.tracker.update(detections)
            license_plates = self.detector.detect_license_plates(image)
            detection_results = []
            frame_number = 1
            for license_plate in license_plates:
                x1, y1, x2, y2, score, class_id = license_plate
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
                if car_id != -1:
                    try:
                        license_plate_crop = image[int(y1):int(y2), int(x1):int(x2)]
                        if license_plate_crop.size > 0:
                            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                            plate_text, text_score = read_license_plate(license_plate_crop_thresh)
                            if plate_text is not None:
                                detection_id = str(uuid.uuid4())
                                detection_result = self.save_detection(
                                    detection_id=detection_id,
                                    frame_number=frame_number,
                                    vehicle_crop=image[int(ycar1):int(ycar2), int(xcar1):int(xcar2)],
                                    plate_crop=license_plate_crop,
                                    plate_text=plate_text,
                                    text_score=text_score
                                )
                                if detection_result:
                                    detection_results.append(detection_result)
                    except Exception as plate_error:
                        print(f"Error processing plate in image: {str(plate_error)}")
                        continue
            return {
                'success': True,
                'message': 'Image processed successfully',
                'detections': detection_results
            }
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return {
                'success': False,
                'message': f'Error processing image: {str(e)}'
            }
