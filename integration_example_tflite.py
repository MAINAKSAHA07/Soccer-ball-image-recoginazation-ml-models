# TensorFlow Lite Integration Example (Python)
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
