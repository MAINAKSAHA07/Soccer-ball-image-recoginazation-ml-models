# Python Integration Example
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
