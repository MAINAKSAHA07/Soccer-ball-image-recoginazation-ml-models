#!/usr/bin/env python3
"""
Test the exported TorchScript model for ball detection
"""

import torch
import cv2
import numpy as np
import os
from pathlib import Path

def test_torchscript_model(model_path: str = 'ball_detection_model.torchscript'):
    """
    Test the TorchScript model with sample images
    
    Args:
        model_path: Path to the TorchScript model
    """
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
    
    print(f"🧪 Testing TorchScript model: {model_path}")
    
    try:
        # Load the model
        print("📥 Loading model...")
        model = torch.jit.load(model_path)
        model.eval()
        
        # Create a test image (random noise for testing)
        print("🖼️ Creating test image...")
        test_image = torch.randn(1, 3, 512, 512)
        
        # Run inference
        print("🔍 Running inference...")
        with torch.no_grad():
            output = model(test_image)
        
        print(f"✅ Model inference successful!")
        print(f"📊 Output shape: {output.shape}")
        print(f"📊 Output type: {type(output)}")
        
        # Test with a real image if available
        test_images = list(Path('test').glob('*.jpg')) + list(Path('test').glob('*.png'))
        
        if test_images:
            print(f"\n🖼️ Testing with real image: {test_images[0]}")
            
            # Load and preprocess image
            image = cv2.imread(str(test_images[0]))
            image = cv2.resize(image, (512, 512))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))  # HWC to CHW
            image = np.expand_dims(image, axis=0)  # Add batch dimension
            image_tensor = torch.from_numpy(image)
            
            # Run inference
            with torch.no_grad():
                real_output = model(image_tensor)
            
            print(f"✅ Real image inference successful!")
            print(f"📊 Real output shape: {real_output.shape}")
            
            # Basic output analysis
            if len(real_output.shape) == 3:
                print(f"📊 Detections: {real_output.shape[2]} potential objects")
                
                # Find detections above threshold
                detections = real_output[0]  # Remove batch dimension
                confidence_scores = detections[4, :]  # Assuming confidence is in 5th row
                high_conf_detections = (confidence_scores > 0.5).sum().item()
                print(f"📊 High confidence detections (>0.5): {high_conf_detections}")
        
        print(f"\n🎉 TorchScript model test completed successfully!")
        print(f"📁 Model file: {model_path}")
        print(f"📏 Model size: {os.path.getsize(model_path) / (1024 * 1024):.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def show_model_info(model_path: str = 'ball_detection_model.torchscript'):
    """Show detailed information about the model"""
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
    
    print(f"\n📋 Model Information:")
    print(f"  - File: {model_path}")
    print(f"  - Size: {os.path.getsize(model_path) / (1024 * 1024):.1f} MB")
    print(f"  - Format: TorchScript (PyTorch)")
    
    try:
        model = torch.jit.load(model_path)
        print(f"  - Model loaded successfully")
        
        # Get model code
        print(f"\n🔧 Model Code:")
        print(model.code)
        
    except Exception as e:
        print(f"  - Error loading model: {e}")

def main():
    """
    Main test function
    """
    print("🧪 Starting TorchScript model testing...")
    
    # Test the model
    success = test_torchscript_model()
    
    if success:
        # Show model information
        show_model_info()
        
        print(f"\n✅ Your TorchScript model is ready for use!")
        print(f"\n📚 Usage options:")
        print(f"1. Use directly in Python with PyTorch")
        print(f"2. Convert to CoreML using Xcode or alternative methods")
        print(f"3. Use with PyTorch Mobile for Android/iOS")
        print(f"4. Deploy on edge devices")
    else:
        print(f"\n❌ Model testing failed. Please check the model file.")

if __name__ == "__main__":
    main() 