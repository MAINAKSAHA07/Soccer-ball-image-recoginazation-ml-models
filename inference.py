#!/usr/bin/env python3
"""
Ball detection inference script for easy integration
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
from typing import List, Tuple, Optional
import torch

class BallDetector:
    """
    Lightweight ball detection class for easy integration
    """
    
    def __init__(self, model_path: str = 'ball_detection_model.pt', confidence: float = 0.5):
        """
        Initialize the ball detector
        
        Args:
            model_path: Path to the trained model
            confidence: Confidence threshold for detections
        """
        self.confidence = confidence
        
        # Load the model
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"✅ Loaded trained model from {model_path}")
        else:
            print(f"⚠️  Trained model not found at {model_path}, using pre-trained YOLOv8n")
            self.model = YOLO('yolov8n.pt')
        
        # Check device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Using device: {self.device}")
    
    def detect_balls(self, image: np.ndarray) -> List[dict]:
        """
        Detect balls in an image
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detection dictionaries with keys:
            - bbox: [x1, y1, x2, y2] bounding box coordinates
            - confidence: Detection confidence score
            - class_id: Class ID (0 for ball)
        """
        # Run inference
        results = self.model(image, conf=self.confidence, verbose=False)
        
        detections = []
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get confidence score
                    conf = box.conf[0].cpu().numpy()
                    
                    # Get class ID
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class_id': class_id
                    })
        
        return detections
    
    def detect_balls_in_image(self, image_path: str) -> Tuple[np.ndarray, List[dict]]:
        """
        Detect balls in an image file
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (image, detections)
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Detect balls
        detections = self.detect_balls(image)
        
        return image, detections
    
    def draw_detections(self, image: np.ndarray, detections: List[dict]) -> np.ndarray:
        """
        Draw detection boxes on the image
        
        Args:
            image: Input image
            detections: List of detections from detect_balls()
            
        Returns:
            Image with detection boxes drawn
        """
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
    
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> str:
        """
        Process a video file and detect balls in each frame
        
        Args:
            video_path: Path to input video
            output_path: Path to output video (optional)
            
        Returns:
            Path to the output video
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set up output video
        if output_path is None:
            base_name = os.path.splitext(video_path)[0]
            output_path = f"{base_name}_detected.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        print(f"🎥 Processing video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect balls in frame
            detections = self.detect_balls(frame)
            
            # Draw detections
            result_frame = self.draw_detections(frame, detections)
            
            # Write frame to output video
            out.write(result_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:  # Print progress every 30 frames
                print(f"Processed {frame_count} frames...")
        
        # Clean up
        cap.release()
        out.release()
        
        print(f"✅ Video processing completed! Output saved to: {output_path}")
        return output_path

def main():
    """
    Example usage of the BallDetector class
    """
    # Initialize detector
    detector = BallDetector(confidence=0.5)
    
    # Example: Detect balls in test images
    test_dir = "test"
    if os.path.exists(test_dir):
        print(f"\n🔍 Testing on {test_dir} directory...")
        
        # Get first few test images
        test_images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][:5]
        
        for image_file in test_images:
            image_path = os.path.join(test_dir, image_file)
            print(f"\nProcessing: {image_file}")
            
            try:
                # Detect balls
                image, detections = detector.detect_balls_in_image(image_path)
                
                print(f"Found {len(detections)} ball(s)")
                for i, detection in enumerate(detections):
                    print(f"  Ball {i+1}: confidence={detection['confidence']:.3f}")
                
                # Draw and save result
                result_image = detector.draw_detections(image, detections)
                output_path = f"result_{image_file}"
                cv2.imwrite(output_path, result_image)
                print(f"Result saved to: {output_path}")
                
            except Exception as e:
                print(f"Error processing {image_file}: {e}")

if __name__ == "__main__":
    main() 