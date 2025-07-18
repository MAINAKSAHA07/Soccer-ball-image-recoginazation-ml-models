#!/usr/bin/env python3
"""
Test different confidence thresholds to find optimal detection
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

def test_confidence_thresholds(video_path, confidence_levels=[0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]):
    """Test different confidence thresholds on a video"""
    
    print("🔍 Testing Confidence Thresholds")
    print("=" * 50)
    
    # Load model
    model_path = 'ball_detection_model.pt'
    if os.path.exists(model_path):
        model = YOLO(model_path)
        print(f"✅ Loaded trained model from {model_path}")
    else:
        print(f"⚠️  Trained model not found at {model_path}, using pre-trained YOLOv8n")
        model = YOLO('yolov8n.pt')
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎥 Video has {total_frames} frames")
    
    results = {}
    
    for conf in confidence_levels:
        print(f"\n🎯 Testing confidence threshold: {conf}")
        
        # Reset video to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        detection_count = 0
        frame_count = 0
        
        # Test first 30 frames for speed
        test_frames = min(30, total_frames)
        
        while frame_count < test_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            detection_results = model(frame, conf=conf, verbose=False)
            
            for result in detection_results:
                if result.boxes is not None and len(result.boxes) > 0:
                    detection_count += 1
                    break  # Found at least one detection in this frame
            
            frame_count += 1
        
        detection_rate = (detection_count / test_frames) * 100
        results[conf] = {
            'detections': detection_count,
            'frames_tested': test_frames,
            'detection_rate': detection_rate
        }
        
        print(f"  📊 Found detections in {detection_count}/{test_frames} frames ({detection_rate:.1f}%)")
    
    cap.release()
    
    # Print summary
    print(f"\n📋 Confidence Threshold Analysis Summary")
    print("=" * 50)
    print(f"{'Confidence':<12} {'Detections':<12} {'Rate':<10}")
    print("-" * 50)
    
    for conf in confidence_levels:
        result = results[conf]
        print(f"{conf:<12} {result['detections']:<12} {result['detection_rate']:<10.1f}%")
    
    # Recommend optimal threshold
    best_conf = max(results.keys(), key=lambda x: results[x]['detection_rate'])
    best_rate = results[best_conf]['detection_rate']
    
    print(f"\n💡 Recommendation:")
    print(f"   Best confidence threshold: {best_conf}")
    print(f"   Detection rate: {best_rate:.1f}%")
    
    if best_rate > 0:
        print(f"   ✅ Use confidence threshold of {best_conf} for best detection")
    else:
        print(f"   ⚠️  No detections found even with lowest confidence")
        print(f"   💡 Consider retraining the model with more diverse data")

if __name__ == '__main__':
    video_path = "/Users/mainaksaha/Desktop/MASTERS/Project/Soccer ball image recoginazation ml models/IMG_4749.MOV"
    
    if os.path.exists(video_path):
        test_confidence_thresholds(video_path)
    else:
        print(f"❌ Video file not found: {video_path}")
        print("Please update the video_path variable in the script") 