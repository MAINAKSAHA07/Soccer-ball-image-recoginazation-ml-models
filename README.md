# ⚽ Soccer Ball Detection Web Application

A real-time soccer ball detection web application built with Flask, YOLOv8, and deployed on Google Cloud Run. The application can detect soccer balls in uploaded videos and provides detailed analysis with confidence scores.

## 🚀 Live Demo

**Application URL**: https://ball-detection-app-auspum45ka-uc.a.run.app

## ✨ Features

- **🎥 Video Upload & Analysis**: Upload video files and get frame-by-frame ball detection
- **🎯 Real-time Detection**: Uses custom-trained YOLOv8 model for accurate ball detection
- **📊 Detailed Analytics**: Shows detection rate, confidence scores, and frame-by-frame results
- **🖼️ Visual Results**: Displays only frames where balls are detected with bounding boxes
- **☁️ Cloud Deployment**: Fully deployed on Google Cloud Run for scalability
- **📱 Responsive UI**: Modern, mobile-friendly interface with drag-and-drop upload

## 🏗️ Architecture

```
├── Frontend (Flask + Bootstrap)
│   ├── Video upload interface
│   ├── Real-time processing display
│   └── Results visualization
├── Backend (Python + Flask)
│   ├── YOLOv8 model inference
│   ├── Video processing pipeline
│   └── REST API endpoints
└── Cloud Infrastructure (Google Cloud Run)
    ├── Containerized deployment
    ├── Auto-scaling
    └── HTTPS endpoint
```

## 🛠️ Technology Stack

- **Backend**: Python 3.11, Flask, OpenCV, Ultralytics YOLOv8
- **Frontend**: HTML5, CSS3, Bootstrap 5, JavaScript
- **ML Model**: Custom-trained YOLOv8 model (`ball_detection_model.pt`)
- **Cloud**: Google Cloud Run, Google Container Registry
- **Deployment**: Docker, Cloud Build

## 📁 Project Structure

```
soccer-ball-detection/
├── app.py                          # Main Flask application
├── ball_detection_model.pt         # Custom-trained YOLOv8 model
├── web_requirements.txt            # Python dependencies
├── Dockerfile                      # Container configuration
├── manual_deploy.sh               # Deployment script
├── templates/
│   └── index.html                 # Web interface
├── debug_video.py                 # Debug script for video analysis
├── test_confidence.py             # Confidence threshold testing
└── README.md                      # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker
- Google Cloud SDK (for deployment)
- Git

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd soccer-ball-detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv web_env
   source web_env/bin/activate  # On Windows: web_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r web_requirements.txt
   ```

4. **Run locally**
   ```bash
   python app.py
   ```

5. **Access the application**
   - Open http://localhost:5000 in your browser
   - Upload a video file to test ball detection

### Cloud Deployment

1. **Set up Google Cloud**
   ```bash
   gcloud auth login
   gcloud config set project ball-detection-app
   ```

2. **Deploy to Cloud Run**
   ```bash
   ./manual_deploy.sh
   ```

3. **Access live application**
   - Visit: https://ball-detection-app-auspum45ka-uc.a.run.app

## 🎯 Model Information

### Training Details
- **Model**: YOLOv8 (Ultralytics)
- **Dataset**: Custom soccer ball dataset
- **Classes**: 1 (soccer ball)
- **Confidence Threshold**: 0.01 (1%)
- **Detection Rate**: ~33% on test videos

### Performance
- **Average Detection Rate**: 33.3%
- **Confidence Range**: 0.08 - 0.24
- **Processing Speed**: ~30 fps on Cloud Run
- **Supported Formats**: MP4, MOV, AVI, etc.

## 🔧 Configuration

### Confidence Threshold
The model uses a confidence threshold of 0.01 (1%) for optimal detection. This can be adjusted in `app.py`:

```python
class BallDetector:
    def __init__(self, model_path='ball_detection_model.pt', confidence=0.01):
```

### Video Processing
- **Max File Size**: 16MB (Cloud Run limit)
- **Supported Formats**: MP4, MOV, AVI, MKV
- **Frame Sampling**: Processes all frames
- **Output Limit**: Shows up to 10 detection frames

## 📊 API Endpoints

### POST `/upload_video`
Upload and process a video file for ball detection.

**Request**: Multipart form data with video file
**Response**: JSON with detection results

```json
{
  "success": true,
  "preview_frames": [...],
  "video_info": {
    "fps": 30,
    "width": 1920,
    "height": 1080,
    "total_frames": 111,
    "frames_processed": 111,
    "detection_rate": 4.5
  },
  "message": "Found balls in 5 out of 111 frames (4.5% detection rate)"
}
```

### GET `/test_model`
Test the model with a simple synthetic image.

**Response**: Model status and test results

## 🧪 Testing & Debugging

### Debug Video Analysis
```bash
python debug_video.py "path/to/video.mp4" --confidence 0.01
```

### Test Confidence Thresholds
```bash
python test_confidence.py
```

### Local Model Testing
```bash
python -c "
from app import detector
import cv2
import numpy as np

# Create test image
test_img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.circle(test_img, (320, 240), 50, (0, 255, 0), -1)

# Test detection
detections = detector.detect_balls(test_img)
print(f'Detections: {len(detections)}')
"
```

## 🐳 Docker

### Build Image
```bash
docker build -t ball-detection-app .
```

### Run Container
```bash
docker run -p 5000:5000 ball-detection-app
```

### Push to Registry
```bash
docker tag ball-detection-app gcr.io/ball-detection-app/ball-detection-app:latest
docker push gcr.io/ball-detection-app/ball-detection-app:latest
```

## ☁️ Cloud Deployment

### Google Cloud Run
The application is deployed on Google Cloud Run for:
- **Auto-scaling**: Scales to zero when not in use
- **Cost optimization**: Pay only for actual usage
- **Global availability**: HTTPS endpoint worldwide
- **Easy updates**: Simple deployment process

### Deployment Process
1. **Build Docker image**
2. **Push to Container Registry**
3. **Deploy to Cloud Run**
4. **Configure HTTPS endpoint**

### Environment Variables
- `PORT`: 5000 (default)
- `UPLOAD_FOLDER`: /app/uploads
- `MAX_CONTENT_LENGTH`: 16MB

## 📈 Performance Optimization

### Model Optimization
- **Confidence Threshold**: Tuned to 0.01 for maximum detection
- **Frame Processing**: Processes all frames for accuracy
- **Memory Management**: Efficient video handling

### Cloud Optimization
- **Container Size**: Optimized Docker image (~856MB)
- **Cold Start**: ~30 seconds for first request
- **Warm Start**: ~2 seconds for subsequent requests

## 🔍 Troubleshooting

### Common Issues

1. **No balls detected**
   - Check video quality and lighting
   - Verify video contains soccer balls
   - Try different confidence thresholds

2. **Upload fails**
   - Ensure file size < 16MB
   - Check video format compatibility
   - Verify network connection

3. **Slow processing**
   - Video length affects processing time
   - Cloud Run has CPU/memory limits
   - Consider shorter test videos

### Debug Tools
- **Debug Script**: `debug_video.py` for detailed analysis
- **Confidence Test**: `test_confidence.py` for threshold optimization
- **Model Test**: `/test_model` endpoint for model verification

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ultralytics**: YOLOv8 framework
- **OpenCV**: Computer vision library
- **Flask**: Web framework
- **Google Cloud**: Cloud infrastructure
- **Bootstrap**: UI framework

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review debug scripts
3. Test with different videos
4. Contact the development team

---

**Built with ❤️ for soccer ball detection enthusiasts** 