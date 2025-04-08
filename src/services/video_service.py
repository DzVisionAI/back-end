import cv2
import numpy as np
from datetime import datetime
from src import db
from src.models.license_plates_model import LicensePlate
from src.models.events_model import Event
import os
from werkzeug.utils import secure_filename
import uuid

class VideoService:
    def __init__(self):
        # Initialize YOLO models
        self.plate_detector = cv2.dnn.readNetFromDarknet(
            'src/models/yolo/alpr_detector.cfg',
            'src.models/yolo/alpr_detector.weights'
        )
        self.plate_recognizer = cv2.dnn.readNetFromDarknet(
            'src/models/yolo/alpr_recognizer.cfg',
            'src.models/yolo/alpr_recognizer.weights'
        )
        
        # Get output layer names
        self.layer_names = self.plate_detector.getLayerNames()
        self.output_layers = [self.layer_names[i-1] for i in self.plate_detector.getUnconnectedOutLayers()]
        
        # Create upload directory if it doesn't exist
        self.upload_folder = 'uploads/videos'
        os.makedirs(self.upload_folder, exist_ok=True)

    def process_video(self, video_file):
        try:
            # Save the uploaded file
            filename = secure_filename(video_file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(self.upload_folder, unique_filename)
            video_file.save(file_path)

            # Open the video file
            cap = cv2.VideoCapture(file_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps

            # Process each frame
            frame_number = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process every 5th frame to reduce processing time
                if frame_number % 5 == 0:
                    # Detect license plates
                    plates = self.detect_plates(frame)
                    
                    # Process each detected plate
                    for plate in plates:
                        # Recognize plate number
                        plate_number = self.recognize_plate(plate)
                        
                        if plate_number:
                            # Save to database
                            self.save_plate_detection(plate_number, frame_number, frame)

                frame_number += 1

            cap.release()
            
            return {
                'success': True,
                'message': 'Video processed successfully',
                'duration': duration,
                'frame_count': frame_count,
                'fps': fps
            }

        except Exception as e:
            return {
                'success': False,
                'message': str(e)
            }

    def detect_plates(self, frame):
        height, width = frame.shape[:2]
        
        # Prepare image for YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.plate_detector.setInput(blob)
        outs = self.plate_detector.forward(self.output_layers)
        
        # Process detections
        plates = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5:  # Confidence threshold
                    # Get coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    # Extract plate region
                    plate = frame[y:y+h, x:x+w]
                    plates.append(plate)
        
        return plates

    def recognize_plate(self, plate):
        # Preprocess plate image
        plate = cv2.resize(plate, (416, 416))
        plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        plate = cv2.threshold(plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Prepare for YOLO
        blob = cv2.dnn.blobFromImage(plate, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.plate_recognizer.setInput(blob)
        outs = self.plate_recognizer.forward()
        
        # Process recognition results
        # This is a simplified version - you'll need to implement proper OCR
        # based on your specific YOLO model's output format
        plate_number = "ABC123"  # Placeholder
        
        return plate_number

    def save_plate_detection(self, plate_number, frame_number, frame):
        try:
            # Save the frame image
            image_path = f'uploads/plates/{plate_number}_{frame_number}.jpg'
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            cv2.imwrite(image_path, frame)
            
            # Create license plate record
            license_plate = LicensePlate(
                plateNumber=plate_number,
                detectedAt=datetime.utcnow(),
                image=image_path
            )
            
            # Create event record
            event = Event(
                typeName='plate_detection',
                description=f'License plate {plate_number} detected',
                time=datetime.utcnow()
            )
            
            # Save to database
            db.session.add(license_plate)
            db.session.add(event)
            db.session.commit()
            
        except Exception as e:
            db.session.rollback()
            raise e 