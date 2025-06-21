import tensorflow as tf
import numpy as np

# Path to your best trained model
model_path = 'models/MobileNetV2/mobilenet_model_ft.keras' # Or your desired model path

# Load the model
try:
    model = tf.keras.models.load_model(model_path)
    print(f"Successfully loaded model from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    # Handle the error, e.g., exit or use a fallback

model.summary()