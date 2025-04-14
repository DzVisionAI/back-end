from ultralytics import YOLO
import cv2
import numpy as np
import csv
import os
import time
import string
import easyocr
import sys
sys.path.append(os.path.abspath("sort"))
from sort import Sort
from collections import defaultdict

# Initialize EasyOCR reader (CPU only)
reader = easyocr.Reader(['en'], gpu=False)

# Character mapping dictionaries
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

class LicensePlateProcessor:
    def __init__(self):
        print("Initializing models (using CPU)...")
        # Initialize models
        self.vehicle_model = YOLO('yolov8n.pt').to('cpu')
        self.plate_model = YOLO('license_plate_detector.pt').to('cpu')
        
        # Optimized tracker settings
        self.tracker = Sort(max_age=10, min_hits=2, iou_threshold=0.3)
        self.frame_count = 0
        self.results = defaultdict(dict)
        self.vehicle_colors = {}
        self.csv_file = './results.csv'
        self.final_csv_file = './finalresults.csv'
        self._init_csv()
        print("Initialization complete.")

    def _init_csv(self):
        """Initialize CSV files with headers"""
        with open(self.csv_file, 'w') as f:
            f.write('frame_nmr,car_id,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score,car_color\n')
        with open(self.final_csv_file, 'w') as f:
            f.write('car_id,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score,car_color\n')

    def _append_to_csv(self, data):
        """Append detection results to CSV"""
        with open(self.csv_file, 'a') as f:
            f.write(','.join(map(str, data)) + '\n')

    def get_dominant_color(self, frame, bbox):
        """Get dominant color from vehicle ROI"""
        x1, y1, x2, y2 = map(int, bbox)
        # Sample every 5th pixel for faster processing
        roi = frame[y1:y2:5, x1:x2:5]
        if roi.size == 0:
            return "unknown"
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Simple histogram-based color detection
        hue = hsv[:,:,0].mean()
        if hue < 15 or hue > 165: return "red"
        elif 15 <= hue < 45: return "yellow"
        elif 45 <= hue < 75: return "green"
        elif 75 <= hue < 105: return "cyan"
        elif 105 <= hue < 135: return "blue"
        elif 135 <= hue < 165: return "magenta"
        else: return "unknown"

    def license_complies_format(self, text):
        """Validate license plate format"""
        if len(text) != 7:
            return False
        for i, char in enumerate(text):
            if i in [0, 1, 4, 5, 6]:  # Should be letters
                if not (char in string.ascii_uppercase or char in dict_int_to_char):
                    return False
            elif i in [2, 3]:  # Should be numbers
                if not (char in string.digits or char in dict_char_to_int):
                    return False
        return True

    def format_license(self, text):
        """Format license plate using mapping dictionaries"""
        formatted = []
        for i, char in enumerate(text):
            if i in [0, 1, 4, 5, 6] and char in dict_int_to_char:
                formatted.append(dict_int_to_char[char])
            elif i in [2, 3] and char in dict_char_to_int:
                formatted.append(dict_char_to_int[char])
            else:
                formatted.append(char)
        return ''.join(formatted)

    def read_license_plate(self, plate_img):
        """Read license plate using EasyOCR"""
        if plate_img.size == 0:
            return None, None
            
        # Convert to grayscale and enhance contrast
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Use EasyOCR to read text
        results = reader.readtext(thresh)
        for result in results:
            text = result[1].upper().replace(' ', '')
            score = result[2]
            if self.license_complies_format(text):
                return self.format_license(text), score
        return None, None

    def get_car(self, plate_bbox, vehicle_track_ids):
        """Match license plate to vehicle"""
        x1, y1, x2, y2, score, _ = plate_bbox
        for vehicle in vehicle_track_ids:
            xcar1, ycar1, xcar2, ycar2, car_id = vehicle
            if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
                return xcar1, ycar1, xcar2, ycar2, car_id
        return -1, -1, -1, -1, -1

    def safe_crop(self, frame, y1, y2, x1, x2):
        """Safely crop image with boundary checks"""
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def process_video(self, video_path):
        """Main processing function"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video file")
            return

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create output video with compatible codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Changed from h264 to more compatible codec
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

        print("Starting video processing...")
        start_time = time.time()
        processed_frames = 0
        total_vehicles = set()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1
            if self.frame_count % 5 != 0:  # Process every 5th frame
                continue

            processed_frames += 1
            
            # Detect vehicles
            vehicle_detections = []
            vehicle_results = self.vehicle_model(frame, imgsz=480, verbose=False)[0]  # Reduced imgsz
            for box in vehicle_results.boxes.data.tolist():
                x1, y1, x2, y2, score, cls = box
                if int(cls) in [2, 3, 5, 7] and score > 0.5:  # Car, bike, bus, truck
                    vehicle_detections.append([x1, y1, x2, y2, score])
            
            # Track vehicles
            tracked_vehicles = self.tracker.update(np.array(vehicle_detections))
            
            # Update total vehicle count
            for vehicle in tracked_vehicles:
                car_id = int(vehicle[4])
                total_vehicles.add(car_id)
                
                # Get and store vehicle color if not already stored
                if car_id not in self.vehicle_colors:
                    bbox = vehicle[:4]
                    self.vehicle_colors[car_id] = self.get_dominant_color(frame, bbox)
            
            # Detect license plates
            plate_results = self.plate_model(frame, imgsz=480, verbose=False)[0]  # Reduced imgsz
            for box in plate_results.boxes.data.tolist():
                x1, y1, x2, y2, score, _ = box
                if score < 0.6:  # Minimum confidence
                    continue
                
                # Match plate to vehicle
                xcar1, ycar1, xcar2, ycar2, car_id = self.get_car(
                    [x1, y1, x2, y2, score, 0], tracked_vehicles
                )
                if car_id == -1:
                    continue
                
                # Safely crop license plate
                plate_crop = self.safe_crop(frame, y1, y2, x1, x2)
                if plate_crop is None:
                    continue
                
                # Read license plate
                plate_text, plate_score = self.read_license_plate(plate_crop)
                if plate_text:
                    # Store results
                    self.results[car_id][self.frame_count] = {
                        'car_bbox': [xcar1, ycar1, xcar2, ycar2],
                        'plate_bbox': [x1, y1, x2, y2],
                        'plate_score': score,
                        'plate_text': plate_text,
                        'text_score': plate_score,
                        'car_color': self.vehicle_colors.get(car_id, "unknown")
                    }
                    
                    # Write to CSV
                    self._append_to_csv([
                        self.frame_count, car_id,
                        f"[{xcar1} {ycar1} {xcar2} {ycar2}]",
                        f"[{x1} {y1} {x2} {y2}]",
                        f"{score:.2f}", plate_text, f"{plate_score:.2f}",
                        self.vehicle_colors.get(car_id, "unknown")
                    ])

            # Write frame to output
            out.write(frame)
            
            # Display processing info
            elapsed = time.time() - start_time
            fps = processed_frames / elapsed if elapsed > 0 else 0
            status = f"Frame: {self.frame_count} | Vehicles: {len(total_vehicles)} | FPS: {fps:.1f}"
            cv2.putText(frame, status, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show preview (optional)
            preview = cv2.resize(frame, (1280, 720))
            cv2.imshow('License Plate Detection', preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Generate final results
        self._generate_final_results()
        
        # Clean up
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Processing complete. Total vehicles detected: {len(total_vehicles)}")
        print(f"Detailed results saved to {self.csv_file}")
        print(f"Final results saved to {self.final_csv_file}")

    def _generate_final_results(self):
        """Generate final results with best detection for each vehicle"""
        with open(self.final_csv_file, 'w') as f:
            f.write('car_id,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score,car_color\n')
            
            for car_id, frames in self.results.items():
                if not frames:
                    continue
                
                # Find frame with highest plate_score
                best_frame = max(frames.items(), key=lambda x: x[1]['plate_score'])
                frame_data = best_frame[1]
                
                # Find best text score for this car_id
                best_text_frame = max(frames.items(), key=lambda x: x[1]['text_score'])
                best_text = best_text_frame[1]['plate_text']
                best_text_score = best_text_frame[1]['text_score']
                
                # Write to final CSV
                f.write(','.join([
                    str(car_id),
                    f"[{' '.join(map(str, frame_data['car_bbox']))}]",
                    f"[{' '.join(map(str, frame_data['plate_bbox']))}]",
                    f"{frame_data['plate_score']:.2f}",
                    best_text,
                    f"{best_text_score:.2f}",
                    frame_data['car_color']
                ]) + '\n')

if __name__ == "__main__":
    processor = LicensePlateProcessor()
    processor.process_video('./sample.mp4')