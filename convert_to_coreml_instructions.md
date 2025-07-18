# Converting TorchScript to CoreML (.mlmodel)

Since we're having compatibility issues with `coremltools` on Python 3.13, here are alternative methods to convert your TorchScript model to CoreML format:

## Method 1: Using Xcode (Recommended)

1. **Open Xcode** and create a new iOS project
2. **Drag and drop** your `ball_detection_model.torchscript` file into the project
3. **Use Xcode's built-in CoreML conversion**:
   - In Xcode, go to `File > Add Files to [Project]`
   - Select your `.torchscript` file
   - Xcode will automatically offer to convert it to CoreML format

## Method 2: Using Python with Conda

If you have conda installed, you can create a separate environment with Python 3.9:

```bash
# Create a new conda environment with Python 3.9
conda create -n coreml_env python=3.9
conda activate coreml_env

# Install coremltools
pip install coremltools==6.3

# Run the conversion script
python convert_onnx_to_coreml.py
```

## Method 3: Using Online Converters

Several online services can convert PyTorch models to CoreML:
- **Apple's Model Converter** (if available)
- **Hugging Face Model Hub** with CoreML support

## Method 4: Manual Conversion Script

Create a new Python environment with Python 3.9:

```bash
# Create virtual environment with Python 3.9
python3.9 -m venv coreml_env
source coreml_env/bin/activate

# Install dependencies
pip install torch torchvision coremltools==6.3

# Copy your model and run conversion
cp ball_detection_model.torchscript coreml_env/
cd coreml_env
python convert_onnx_to_coreml.py
```

## Current Status

✅ **Successfully exported**: `ball_detection_model.torchscript` (11.8 MB)
- This is a fully functional PyTorch model
- Can be used directly in Python applications
- Ready for conversion to CoreML using any of the methods above

## Model Information

- **Input shape**: (1, 3, 512, 512) - RGB images at 512x512 resolution
- **Output shape**: (1, 5, 5376) - Detection results with bounding boxes
- **Model size**: 11.8 MB (compressed)
- **Framework**: PyTorch TorchScript (optimized for production)

## Usage in iOS/macOS

Once you have the `.mlmodel` file, you can use it in your iOS/macOS app:

```swift
import CoreML
import Vision

// Load the model
guard let model = try? VNCoreMLModel(for: BallDetectionModel().model) else {
    fatalError("Failed to load CoreML model")
}

// Create a request
let request = VNCoreMLRequest(model: model) { request, error in
    // Handle results
    guard let results = request.results as? [VNRecognizedObjectObservation] else { return }
    
    for observation in results {
        let boundingBox = observation.boundingBox
        let confidence = observation.confidence
        print("Ball detected at \(boundingBox) with confidence \(confidence)")
    }
}

// Run inference
let handler = VNImageRequestHandler(cgImage: yourImage.cgImage!)
try handler.perform([request])
```

## Next Steps

1. Choose one of the conversion methods above
2. Convert your `ball_detection_model.torchscript` to CoreML format
3. Integrate the `.mlmodel` file into your iOS/macOS project
4. Test the model with your specific use case 