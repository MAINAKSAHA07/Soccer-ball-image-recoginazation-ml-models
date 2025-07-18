#!/usr/bin/env python3
"""
Convert TorchScript model to CoreML format with better error handling
"""

import os
import sys

def check_coremltools_installation():
    """Check if coremltools is properly installed"""
    try:
        import coremltools as ct
        print("✅ coremltools is installed")
        return True
    except ImportError:
        print("❌ coremltools is not installed")
        return False
    except Exception as e:
        print(f"⚠️ coremltools has issues: {e}")
        return False

def convert_torchscript_to_coreml(torchscript_path: str = 'ball_detection_model.torchscript'):
    """
    Convert TorchScript model to CoreML format
    
    Args:
        torchscript_path: Path to the TorchScript model file
    """
    
    if not os.path.exists(torchscript_path):
        print(f"❌ TorchScript model not found at {torchscript_path}")
        print("Please export the model to TorchScript first using export_model.py")
        return False
    
    print(f"🔄 Converting {torchscript_path} to CoreML format...")
    
    try:
        import coremltools as ct
        import torch
        
        # Load the TorchScript model
        print("📥 Loading TorchScript model...")
        torchscript_model = torch.jit.load(torchscript_path)
        
        # Create example input for tracing
        print("🔧 Creating example input...")
        example_input = torch.randn(1, 3, 512, 512)
        
        # Convert to CoreML
        print("🔄 Converting to CoreML...")
        coreml_model = ct.convert(
            torchscript_model,
            inputs=[ct.TensorType(name="input", shape=example_input.shape)],
            minimum_deployment_target=ct.target.iOS15
        )
        
        # Save the CoreML model
        coreml_path = torchscript_path.replace('.torchscript', '.mlpackage')
        coreml_model.save(coreml_path)
        
        print(f"✅ Successfully converted to CoreML: {coreml_path}")
        
        # Print model info
        print(f"\n📊 CoreML Model Info:")
        print(f"  - Input shape: {coreml_model.get_spec().description.input[0].type.multiArrayType.shape}")
        print(f"  - Output shape: {coreml_model.get_spec().description.output[0].type.multiArrayType.shape}")
        print(f"  - File size: {os.path.getsize(coreml_path) / (1024 * 1024):.1f} MB")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 This is likely due to Python 3.13 compatibility issues with coremltools")
        return False
    except Exception as e:
        print(f"❌ Failed to convert TorchScript to CoreML: {e}")
        return False

def provide_alternative_solutions():
    """Provide alternative solutions for CoreML conversion"""
    print("\n🔧 Alternative Solutions:")
    print("\n1️⃣ **Xcode Method (Recommended):**")
    print("   - Open Xcode")
    print("   - Create new iOS project")
    print("   - Drag ball_detection_model.torchscript into project")
    print("   - Xcode will offer CoreML conversion")
    
    print("\n2️⃣ **Conda Environment:**")
    print("   conda create -n coreml_env python=3.9")
    print("   conda activate coreml_env")
    print("   pip install coremltools==6.3")
    print("   python convert_torchscript_to_coreml.py")
    
    print("\n3️⃣ **Python 3.9 Virtual Environment:**")
    print("   python3.9 -m venv coreml_env")
    print("   source coreml_env/bin/activate")
    print("   pip install torch torchvision coremltools==6.3")
    print("   python convert_torchscript_to_coreml.py")
    
    print("\n4️⃣ **Use TorchScript Directly:**")
    print("   - Your .torchscript file works in Python")
    print("   - Can be used with PyTorch Mobile")
    print("   - Compatible with many deployment platforms")

def main():
    """
    Main conversion function
    """
    print("🚀 Starting TorchScript to CoreML conversion...")
    
    # Check coremltools installation
    if not check_coremltools_installation():
        print("\n❌ coremltools is not properly installed or has compatibility issues")
        provide_alternative_solutions()
        return
    
    # Try conversion
    success = convert_torchscript_to_coreml()
    
    if not success:
        print("\n❌ Conversion failed due to compatibility issues")
        provide_alternative_solutions()
    else:
        print("\n🎉 Conversion process completed successfully!")
        print("\n📚 Next steps:")
        print("1. Use the .mlmodel file in your iOS/macOS project")
        print("2. Test the model with your specific use case")
        print("3. Adjust confidence thresholds as needed")

if __name__ == "__main__":
    main() 