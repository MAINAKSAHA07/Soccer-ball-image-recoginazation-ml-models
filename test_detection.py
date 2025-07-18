#!/usr/bin/env python3
"""
Test script to verify ball detection is working
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

def test_ball_detection():
    # Load the model
    model_path = 'ball_detection_model.pt'
    if os.path.exists(model_path):
        model = YOLO(model_path)
        print(f"✅ Loaded trained model from {model_path}")
    else:
        print(f"⚠️  Trained model not found at {model_path}, using pre-trained YOLOv8n")
        model = YOLO('yolov8n.pt')
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    print("🎥 Testing ball detection with webcam...")
    print("Press 'q' to quit, 's' to save a test frame")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run detection every 10 frames to avoid too much output
        if frame_count % 10 == 0:
            try:
                # Run inference
                results = model(frame, conf=0.3, verbose=False)
                
                # Process results
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
                
                print(f"Frame {frame_count}: Detected {len(detections)} objects")
                
                # Draw detections on frame
                for detection in detections:
                    bbox = detection['bbox']
                    confidence = detection['confidence']
                    
                    # Draw bounding box
                    cv2.rectangle(frame, 
                                 (bbox[0], bbox[1]), 
                                 (bbox[2], bbox[3]), 
                                 (0, 255, 0), 2)
                    
                    # Draw confidence score
                    label = f"Ball: {confidence:.2f}"
                    cv2.putText(frame, label, 
                               (bbox[0], bbox[1] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                               (0, 255, 0), 2)
                
            except Exception as e:
                print(f"❌ Error in detection: {e}")
        
        # Display frame
        cv2.imshow('Ball Detection Test', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            cv2.imwrite('test_frame.jpg', frame)
            print("💾 Saved test_frame.jpg")
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Test completed")

if __name__ == '__main__':
    test_ball_detection() 