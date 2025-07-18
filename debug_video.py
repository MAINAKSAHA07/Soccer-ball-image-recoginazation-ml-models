#!/usr/bin/env python3
"""
Debug script to analyze video frame by frame and save detection results
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import argparse
from datetime import datetime

class VideoDebugger:
    def __init__(self, model_path='ball_detection_model.pt', confidence=0.1):
        """
        Initialize the video debugger with lower confidence for testing
        """
        self.confidence = confidence
        print(f"🔧 Using confidence threshold: {confidence}")
        
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"✅ Loaded trained model from {model_path}")
        else:
            print(f"⚠️  Trained model not found at {model_path}, using pre-trained YOLOv8n")
            self.model = YOLO('yolov8n.pt')
    
    def detect_balls(self, image):
        """Enhanced ball detection with detailed logging"""
        try:
            # Ensure image is in the right format
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Run inference with verbose output
            results = self.model(image_rgb, conf=self.confidence, verbose=False)
            detections = []
            
            print(f"📊 Model returned {len(results)} result(s)")
            
            for i, result in enumerate(results):
                print(f"  Result {i}:")
                if result.boxes is not None and len(result.boxes) > 0:
                    print(f"    Found {len(result.boxes)} detection(s)")
                    boxes = result.boxes
                    for j, box in enumerate(boxes):
                        try:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                            
                            print(f"      Box {j}: class={class_id}, conf={conf:.3f}, bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                            
                            # Ensure coordinates are valid
                            if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0:
                                detections.append({
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'confidence': float(conf),
                                    'class_id': class_id
                                })
                        except Exception as box_error:
                            print(f"      ❌ Error processing box {j}: {box_error}")
                            continue
                else:
                    print(f"    No detections found")
            
            return detections
        except Exception as e:
            print(f"❌ Error in ball detection: {e}")
            return []
    
    def draw_detections(self, image, detections, frame_num):
        """Draw detections with detailed information"""
        result_image = image.copy()
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_id = detection['class_id']
            
            # Draw bounding box
            color = (0, 255, 0) if class_id == 0 else (255, 0, 0)  # Green for ball, red for others
            cv2.rectangle(result_image, 
                         (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), 
                         color, 2)
            
            # Draw detailed label
            label = f"Class:{class_id} Conf:{confidence:.3f}"
            cv2.putText(result_image, label, 
                       (bbox[0], bbox[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       color, 2)
        
        # Add frame number
        cv2.putText(result_image, f"Frame: {frame_num}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return result_image
    
    def analyze_video(self, video_path, output_dir="debug_output"):
        """Analyze video frame by frame and save results"""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"🎥 Video Info: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        frame_count = 0
        detection_count = 0
        
        # Create summary file
        summary_file = os.path.join(output_dir, f"analysis_summary_{timestamp}.txt")
        
        with open(summary_file, 'w') as f:
            f.write(f"Video Analysis Summary\n")
            f.write(f"=====================\n")
            f.write(f"Video: {video_path}\n")
            f.write(f"Resolution: {width}x{height}\n")
            f.write(f"FPS: {fps}\n")
            f.write(f"Total Frames: {total_frames}\n")
            f.write(f"Confidence Threshold: {self.confidence}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write(f"Frame Analysis:\n")
            f.write(f"==============\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"\n🔄 Processing Frame {frame_count}/{total_frames}")
            
            # Detect balls
            detections = self.detect_balls(frame)
            
            # Log results
            with open(summary_file, 'a') as f:
                f.write(f"Frame {frame_count}: {len(detections)} detection(s)\n")
                for det in detections:
                    f.write(f"  - Class {det['class_id']}, Conf {det['confidence']:.3f}, BBox {det['bbox']}\n")
            
            # Save frame if detections found
            if len(detections) > 0:
                detection_count += 1
                print(f"🎯 Found {len(detections)} detection(s) in frame {frame_count}")
                
                # Draw detections
                result_frame = self.draw_detections(frame, detections, frame_count)
                
                # Save original and annotated frames
                original_path = os.path.join(output_dir, f"frame_{frame_count:04d}_original.jpg")
                annotated_path = os.path.join(output_dir, f"frame_{frame_count:04d}_detected.jpg")
                
                cv2.imwrite(original_path, frame)
                cv2.imwrite(annotated_path, result_frame)
                
                print(f"💾 Saved: {original_path}")
                print(f"💾 Saved: {annotated_path}")
            else:
                print(f"❌ No detections in frame {frame_count}")
            
            # Save every 10th frame for reference (even without detections)
            if frame_count % 10 == 0:
                ref_path = os.path.join(output_dir, f"frame_{frame_count:04d}_reference.jpg")
                cv2.imwrite(ref_path, frame)
                print(f"📷 Saved reference frame: {ref_path}")
        
        cap.release()
        
        # Final summary
        print(f"\n🎉 Analysis Complete!")
        print(f"📊 Processed {frame_count} frames")
        print(f"🎯 Found detections in {detection_count} frames")
        print(f"📁 Results saved in: {output_dir}")
        print(f"📄 Summary saved as: {summary_file}")
        
        with open(summary_file, 'a') as f:
            f.write(f"\nSummary:\n")
            f.write(f"========\n")
            f.write(f"Total frames processed: {frame_count}\n")
            f.write(f"Frames with detections: {detection_count}\n")
            f.write(f"Detection rate: {detection_count/frame_count*100:.1f}%\n")

def main():
    parser = argparse.ArgumentParser(description='Debug video ball detection')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--confidence', type=float, default=0.1, help='Confidence threshold (default: 0.1)')
    parser.add_argument('--model', default='ball_detection_model.pt', help='Model path')
    parser.add_argument('--output', default='debug_output', help='Output directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"❌ Video file not found: {args.video_path}")
        return
    
    print(f"🔍 Starting video analysis...")
    print(f"📹 Video: {args.video_path}")
    print(f"🎯 Confidence: {args.confidence}")
    print(f"🤖 Model: {args.model}")
    print(f"📁 Output: {args.output}")
    
    debugger = VideoDebugger(args.model, args.confidence)
    debugger.analyze_video(args.video_path, args.output)

if __name__ == '__main__':
    main() 