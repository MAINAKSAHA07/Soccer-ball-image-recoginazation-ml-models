#!/usr/bin/env python3
"""
Export trained ball detection model in various formats
"""

import os
from ultralytics import YOLO
import torch

def export_model(model_path: str = 'ball_detection_model.pt'):
    """
    Export the trained model in multiple formats for easy integration
    
    Args:
        model_path: Path to the trained PyTorch model
    """
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first using train_model.py")
        return
    
    print(f"📦 Exporting model from {model_path}")
    
    # Load the model
    model = YOLO(model_path)
    
    # Export formats
    export_formats = {
        'torchscript': '.torchscript',
        'onnx': '.onnx',
       # 'engine': '.engine',  # TensorRT
        #'coreml': '.mlmodel',  # CoreML for iOS
        #'saved_model': '_saved_model',  # TensorFlow SavedModel
       # 'pb': '.pb',  # TensorFlow GraphDef
       # 'tflite': '.tflite',  # TensorFlow Lite
       # 'edgetpu': '_edgetpu.tflite',  # Edge TPU
       # 'tfjs': '_web_model',  # TensorFlow.js
       # 'paddle': '_paddle_model',  # PaddlePaddle
       # 'ncnn': '_ncnn_model',  # NCNN
       # 'openvino': '_openvino_model',  # OpenVINO
    }
    
    exported_files = []
    
    for format_name, extension in export_formats.items():
        try:
            print(f"🔄 Exporting to {format_name.upper()} format...")
            
            # Export the model
            exported_path = model.export(format=format_name, imgsz=512)
            
            if exported_path and os.path.exists(exported_path):
                exported_files.append((format_name, exported_path))
                print(f"✅ Successfully exported to: {exported_path}")
            else:
                print(f"⚠️  Export to {format_name} failed or file not found")
                
        except Exception as e:
            print(f"❌ Failed to export to {format_name}: {e}")
    
    # Print summary
    print(f"\n📋 Export Summary:")
    print(f"Successfully exported {len(exported_files)} formats:")
    for format_name, file_path in exported_files:
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"  - {format_name.upper()}: {file_path} ({file_size:.1f} MB)")
    
    return exported_files

def create_integration_example():
    """
    Create example integration code for different platforms
    """
    
    print("\n📝 Creating integration examples...")
    
    # Python integration example
    python_example = '''# Python Integration Example
from inference import BallDetector
import cv2

# Initialize detector
detector = BallDetector(model_path='ball_detection_model.pt', confidence=0.5)

# Detect balls in image
image = cv2.imread('your_image.jpg')
detections = detector.detect_balls(image)

# Process results
for detection in detections:
    bbox = detection['bbox']
    confidence = detection['confidence']
    print(f"Ball detected at {bbox} with confidence {confidence:.3f}")

# Draw results
result_image = detector.draw_detections(image, detections)
cv2.imwrite('result.jpg', result_image)
'''
    
    with open('integration_example.py', 'w') as f:
        f.write(python_example)
    
    # ONNX integration example
    onnx_example = '''# ONNX Integration Example (C++)
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// Load ONNX model and run inference
// This is a basic structure - you'll need to implement the full ONNX runtime code
'''
    
    with open('integration_example_onnx.cpp', 'w') as f:
        f.write(onnx_example)
    
    # TensorFlow Lite integration example
    tflite_example = '''# TensorFlow Lite Integration Example (Python)
import tensorflow as tf
import numpy as np
import cv2

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="ball_detection_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input image
image = cv2.imread('your_image.jpg')
image = cv2.resize(image, (512, 512))
image = image.astype(np.float32) / 255.0
image = np.expand_dims(image, axis=0)

# Run inference
interpreter.set_tensor(input_details[0]['index'], image)
interpreter.invoke()

# Get results
detections = interpreter.get_tensor(output_details[0]['index'])
'''
    
    with open('integration_example_tflite.py', 'w') as f:
        f.write(tflite_example)
    
    print("✅ Integration examples created:")
    print("  - integration_example.py (Python)")
    print("  - integration_example_onnx.cpp (C++/ONNX)")
    print("  - integration_example_tflite.py (TensorFlow Lite)")

def main():
    """
    Main export function
    """
    print("🚀 Starting model export process...")
    
    # Export the model
    exported_files = export_model()
    
    # Create integration examples
    create_integration_example()
    
    print("\n🎉 Export process completed!")
    print("\n📚 Next steps:")
    print("1. Use the exported model files in your target platform")
    print("2. Check the integration examples for usage patterns")
    print("3. Adjust confidence thresholds based on your needs")
    print("4. Test the model on your specific use case")

if __name__ == "__main__":
    main() 