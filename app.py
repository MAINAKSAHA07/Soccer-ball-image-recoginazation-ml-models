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
    def __init__(self, model_path='ball_detection_model.pt', confidence=0.01):
        self.confidence = confidence
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"✅ Loaded trained model from {model_path}")
        else:
            print(f"⚠️  Trained model not found at {model_path}, using pre-trained YOLOv8n")
            self.model = YOLO('yolov8n.pt')
    
    def detect_balls(self, image):
        try:
            # Ensure image is in the right format
            if len(image.shape) == 3:
                # Convert BGR to RGB if needed
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Run inference
            results = self.model(image_rgb, conf=self.confidence, verbose=False)
            detections = []
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    for box in boxes:
                        try:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                            
                            # Ensure coordinates are valid
                            if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0:
                                detections.append({
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'confidence': float(conf),
                                    'class_id': class_id
                                })
                        except Exception as box_error:
                            print(f"❌ Error processing detection box: {box_error}")
                            continue
            
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
    
    # Note: Webcam won't work in Cloud Run environment
    # This is a limitation of serverless platforms
    print("⚠️  Webcam is not available in Cloud Run environment")
    print("💡 Use the video upload feature instead")
    
    # Create a placeholder frame with message
    placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder_frame, "Webcam not available", (50, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(placeholder_frame, "in Cloud Run environment", (50, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(placeholder_frame, "Please use video upload", (50, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Convert to JPEG
    _, buffer = cv2.imencode('.jpg', placeholder_frame)
    frame_bytes = buffer.tobytes()
    
    with webcam_lock:
        webcam_frame = frame_bytes
    
    # Keep the thread alive for a while to show the message
    time.sleep(5)
    webcam_active = False

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
            
            print(f"🎥 Processing video: {width}x{height} @ {fps}fps ({total_frames} frames)")
            print(f"🎯 Using confidence threshold: {detector.confidence}")
            
            # Process all frames and collect detections
            detected_frames = []
            frame_count = 0
            detection_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    # Detect balls
                    detections = detector.detect_balls(frame)
                    
                    if len(detections) > 0:
                        detection_count += 1
                        print(f"🎯 Frame {frame_count + 1}: Found {len(detections)} ball(s)")
                        
                        # Draw detections
                        result_frame = detector.draw_detections(frame, detections)
                        
                        # Convert to base64 for preview
                        _, buffer = cv2.imencode('.jpg', result_frame)
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        detected_frames.append({
                            'frame': frame_base64,
                            'detections': detections,
                            'frame_number': frame_count + 1,
                            'confidence_scores': [d['confidence'] for d in detections]
                        })
                        
                        # Limit to first 10 detections for web display
                        if len(detected_frames) >= 10:
                            break
                    
                except Exception as e:
                    print(f"❌ Error processing frame {frame_count + 1}: {e}")
                
                frame_count += 1
                
                # Progress update every 50 frames
                if frame_count % 50 == 0:
                    print(f"📊 Processed {frame_count}/{total_frames} frames...")
            
            cap.release()
            
            # Clean up uploaded file
            os.remove(filepath)
            
            print(f"🎉 Analysis Complete!")
            print(f"📊 Processed {frame_count} frames")
            print(f"🎯 Found detections in {len(detected_frames)} frames")
            
            if len(detected_frames) == 0:
                return jsonify({
                    'success': True,
                    'preview_frames': [],
                    'video_info': {
                        'fps': fps,
                        'width': width,
                        'height': height,
                        'total_frames': total_frames,
                        'frames_processed': frame_count,
                        'detection_rate': 0.0
                    },
                    'message': f'No balls detected in {frame_count} frames. Try a different video or check if the video contains soccer balls.'
                })
            
            detection_rate = (len(detected_frames) / frame_count) * 100
            
            return jsonify({
                'success': True,
                'preview_frames': detected_frames,
                'video_info': {
                    'fps': fps,
                    'width': width,
                    'height': height,
                    'total_frames': total_frames,
                    'frames_processed': frame_count,
                    'detection_rate': round(detection_rate, 2)
                },
                'message': f'Found balls in {len(detected_frames)} out of {frame_count} frames ({detection_rate:.1f}% detection rate)'
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

@app.route('/test_model')
def test_model():
    """Test endpoint to verify model is working"""
    try:
        # Create a test image (simple colored circle)
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(test_image, (320, 240), 50, (0, 255, 0), -1)  # Green circle
        
        # Run detection
        detections = detector.detect_balls(test_image)
        
        return jsonify({
            'success': True,
            'model_loaded': True,
            'test_detections': len(detections),
            'confidence_threshold': detector.confidence,
            'message': f'Model test completed. Found {len(detections)} objects in test image.'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Model test failed'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    app.run(debug=False, host='0.0.0.0', port=port) 