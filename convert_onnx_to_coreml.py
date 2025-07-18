#!/usr/bin/env python3
"""
Convert ONNX model to CoreML format
"""

import os
import coremltools as ct

def convert_onnx_to_coreml(onnx_path: str = 'ball_detection_model.onnx'):
    """
    Convert ONNX model to CoreML format
    
    Args:
        onnx_path: Path to the ONNX model file
    """
    
    if not os.path.exists(onnx_path):
        print(f"❌ ONNX model not found at {onnx_path}")
        print("Please export the model to ONNX first using export_model.py")
        return
    
    print(f"🔄 Converting {onnx_path} to CoreML format...")
    
    try:
        # Load the ONNX model
        onnx_model = ct.converters.onnx.convert(
            model=onnx_path,
            minimum_deployment_target=ct.target.iOS15
        )
        
        # Save the CoreML model
        coreml_path = onnx_path.replace('.onnx', '.mlmodel')
        onnx_model.save(coreml_path)
        
        print(f"✅ Successfully converted to CoreML: {coreml_path}")
        
        # Print model info
        print(f"\n📊 CoreML Model Info:")
        print(f"  - Input shape: {onnx_model.get_spec().description.input[0].type.multiArrayType.shape}")
        print(f"  - Output shape: {onnx_model.get_spec().description.output[0].type.multiArrayType.shape}")
        print(f"  - File size: {os.path.getsize(coreml_path) / (1024 * 1024):.1f} MB")
        
    except Exception as e:
        print(f"❌ Failed to convert ONNX to CoreML: {e}")
        print("\n💡 Troubleshooting tips:")
        print("1. Make sure coremltools is properly installed")
        print("2. Try installing coremltools with: pip install coremltools==6.3")
        print("3. Ensure you're on macOS with Xcode installed")

def main():
    """
    Main conversion function
    """
    print("🚀 Starting ONNX to CoreML conversion...")
    convert_onnx_to_coreml()
    print("\n🎉 Conversion process completed!")

if __name__ == "__main__":
    main() 