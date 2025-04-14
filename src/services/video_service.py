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
                                        self.save_detection(
                                            detection_id=detection_id,
                                            frame_number=frame_number,
                                            vehicle_crop=frame[int(ycar1):int(ycar2), int(xcar1):int(xcar2)],
                                            plate_crop=license_plate_crop,
                                            plate_text=plate_text,
                                            text_score=text_score
                                        )
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
                'total_processing_time': sum(processing_times)
            }

        except Exception as e:
            print(f"Error processing video: {str(e)}")
            return {
                'success': False,
                'message': f'Error processing video: {str(e)}'
            }

    def save_detection(self, detection_id, frame_number, vehicle_crop, plate_crop, plate_text, text_score):
        try:
            # Save the vehicle image
            vehicle_path = os.path.join(self.plates_folder, f'vehicle_{detection_id}_{frame_number}.jpg')
            cv2.imwrite(vehicle_path, vehicle_crop)
            
            # Save the plate image
            plate_path = os.path.join(self.plates_folder, f'plate_{detection_id}_{frame_number}.jpg')
            cv2.imwrite(plate_path, plate_crop)
            
            # Create license plate record
            license_plate = LicensePlate(
                plateNumber=plate_text,
                detectedAt=datetime.utcnow(),
                image=plate_path,
                confidence=float(text_score)
            )
            
            # Create event record
            event = Event(
                typeName='license_plate_detection',
                description=f'License plate {plate_text} detected in frame {frame_number}',
                time=datetime.utcnow()
            )
            
            # Save to database
            db.session.add(license_plate)
            db.session.add(event)
            db.session.commit()
            
        except Exception as e:
            print(f"Error saving detection: {str(e)}")
            db.session.rollback()
            raise e 