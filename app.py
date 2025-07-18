from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import os
import tempfile
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import base64
import io
from PIL import Image
import threading
import time

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for webcam
webcam_active = False
webcam_thread = None
webcam_frame = None
webcam_lock = threading.Lock()

class BallDetector:
    def __init__(self, model_path='ball_detection_model.pt', confidence=0.3):
        self.confidence = confidence
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"✅ Loaded trained model from {model_path}")
        else:
            print(f"⚠️  Trained model not found at {model_path}, using pre-trained YOLOv8n")
            self.model = YOLO('yolov8n.pt')
    
    def detect_balls(self, image):
        try:
            results = self.model(image, conf=self.confidence, verbose=False)
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf),
                            'class_id': class_id
                        })
            
            return detections
        except Exception as e:
            print(f"❌ Error in ball detection: {e}")
            return []
    
    def draw_detections(self, image, detections):
        result_image = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(result_image, 
                         (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), 
                         (0, 255, 0), 2)
            
            # Draw confidence score
            label = f"Ball: {confidence:.2f}"
            cv2.putText(result_image, label, 
                       (bbox[0], bbox[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 255, 0), 2)
        
        return result_image

# Initialize ball detector
detector = BallDetector()

def webcam_worker():
    global webcam_active, webcam_frame
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("🎥 Webcam started successfully")
    
    while webcam_active:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect balls in frame
        detections = detector.detect_balls(frame)
        
        # Debug: Print detection count every 30 frames
        if hasattr(webcam_worker, 'frame_count'):
            webcam_worker.frame_count += 1
        else:
            webcam_worker.frame_count = 0
            
        if webcam_worker.frame_count % 30 == 0:
            print(f"🔍 Detected {len(detections)} balls in frame")
        
        # Draw detections
        result_frame = detector.draw_detections(frame, detections)
        
        # Convert to JPEG
        _, buffer = cv2.imencode('.jpg', result_frame)
        frame_bytes = buffer.tobytes()
        
        with webcam_lock:
            webcam_frame = frame_bytes
        
        time.sleep(0.033)  # ~30 FPS
    
    cap.release()
    print("🎥 Webcam stopped")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process video
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                return jsonify({'error': 'Could not open video file'}), 400
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Process first few frames for preview
            preview_frames = []
            frame_count = 0
            max_preview_frames = 5
            
            while frame_count < max_preview_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect balls
                detections = detector.detect_balls(frame)
                
                # Draw detections
                result_frame = detector.draw_detections(frame, detections)
                
                # Convert to base64 for preview
                _, buffer = cv2.imencode('.jpg', result_frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                preview_frames.append({
                    'frame': frame_base64,
                    'detections': detections
                })
                
                frame_count += 1
            
            cap.release()
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'preview_frames': preview_frames,
                'video_info': {
                    'fps': fps,
                    'width': width,
                    'height': height,
                    'total_frames': total_frames
                }
            })
            
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

@app.route('/start_webcam')
def start_webcam():
    global webcam_active, webcam_thread
    
    if not webcam_active:
        webcam_active = True
        webcam_thread = threading.Thread(target=webcam_worker)
        webcam_thread.start()
        return jsonify({'success': True, 'message': 'Webcam started'})
    else:
        return jsonify({'success': False, 'message': 'Webcam already active'})

@app.route('/stop_webcam')
def stop_webcam():
    global webcam_active
    
    webcam_active = False
    if webcam_thread:
        webcam_thread.join()
    return jsonify({'success': True, 'message': 'Webcam stopped'})

@app.route('/webcam_feed')
def webcam_feed():
    def generate():
        while webcam_active:
            with webcam_lock:
                if webcam_frame:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + webcam_frame + b'\r\n')
            time.sleep(0.033)
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002) 