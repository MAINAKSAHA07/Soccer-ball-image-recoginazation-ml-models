# Soccer Ball Detection Web Application

A modern web application for detecting soccer balls in videos and live webcam feeds using a trained YOLOv8 model.

## Features

- **Video Upload**: Upload video files and get ball detection results with preview frames
- **Live Webcam**: Real-time ball detection using your computer's webcam
- **Modern UI**: Beautiful, responsive interface with drag-and-drop functionality
- **Real-time Processing**: Instant feedback and results display

## Prerequisites

- Python 3.8 or higher
- Webcam (for live detection feature)
- Modern web browser with JavaScript enabled

## Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd "Soccer ball image recoginazation ml models"
   ```

2. **Install the required dependencies**
   ```bash
   pip install -r web_requirements.txt
   ```

3. **Ensure the ball detection model is available**
   - The application expects `ball_detection_model.pt` to be in the root directory
   - If the model is not found, it will fall back to using the pre-trained YOLOv8n model

## Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Open your web browser**
   - Navigate to `http://localhost:5000`
   - The application will be available on all network interfaces

3. **Use the application**
   - **Video Upload**: Drag and drop video files or click to browse
   - **Live Webcam**: Click "Start Webcam" to begin real-time detection

## Usage

### Video Upload
1. Click on the upload area or drag and drop a video file
2. The application will process the first few frames of the video
3. Results will show preview frames with detected balls highlighted
4. Each frame displays the number of balls detected

### Live Webcam
1. Click "Start Webcam" to activate your camera
2. The application will show real-time ball detection
3. Detected balls will be highlighted with green bounding boxes
4. Click "Stop Webcam" to end the session

## File Structure

```
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Web interface
├── uploads/              # Temporary upload directory (auto-created)
├── ball_detection_model.pt  # Trained ball detection model
├── web_requirements.txt  # Python dependencies
└── WEB_README.md        # This file
```

## Technical Details

- **Backend**: Flask web framework
- **Model**: YOLOv8-based ball detection model
- **Frontend**: HTML5, CSS3, JavaScript with Bootstrap 5
- **Video Processing**: OpenCV for video handling
- **Real-time Detection**: Threaded webcam processing

## Troubleshooting

### Common Issues

1. **Webcam not working**
   - Ensure your browser has permission to access the camera
   - Check if another application is using the webcam
   - Try refreshing the page

2. **Model not loading**
   - Verify `ball_detection_model.pt` exists in the project directory
   - The application will use a fallback model if the trained model is not found

3. **Video upload fails**
   - Check file size (max 16MB)
   - Ensure the file is a valid video format
   - Try a different video file

4. **Performance issues**
   - Close other applications using the GPU
   - Reduce video resolution for faster processing
   - The application automatically uses CPU if CUDA is not available

### Error Messages

- **"Could not open webcam"**: Camera is not available or in use
- **"No video file provided"**: No file was selected for upload
- **"Could not open video file"**: Invalid or corrupted video file

## Development

### Adding New Features

1. **Modify `app.py`** for backend functionality
2. **Update `templates/index.html`** for frontend changes
3. **Test thoroughly** with different video formats and webcam setups

### Customization

- **Model Path**: Change the model path in the `BallDetector` class
- **Confidence Threshold**: Adjust the confidence parameter for detection sensitivity
- **UI Styling**: Modify the CSS in the HTML template
- **Video Processing**: Customize the number of preview frames in the upload handler

## Security Notes

- The application runs on all network interfaces by default
- For production use, consider:
  - Using HTTPS
  - Implementing user authentication
  - Limiting file upload sizes
  - Adding rate limiting

## License

This project is part of a soccer ball detection system. Please refer to the main project documentation for licensing information. 