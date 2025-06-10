import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image # For image loading utilities

# Define the path to your single image
image_path = 'data/test/Karlskirche.jpg' # IMPORTANT: Change this to your image path

# Define the target size that your model expects (224x224 for MobileNetV2)
img_size = (224, 224)

# 1. Load the image
img = image.load_img(image_path, target_size=img_size)

# 2. Convert the image to a NumPy array
img_array = image.img_to_array(img)

# 3. Rescale the pixel values (same as your ImageDataGenerator's rescale=1./255)
img_array = img_array / 255.0

# 4. Add a batch dimension
# The model expects input shape (batch_size, height, width, channels).
# For a single image, you add an extra dimension at the beginning.
img_array = np.expand_dims(img_array, axis=0) # Now shape will be (1, 224, 224, 3)

print(f"Processed image shape: {img_array.shape}")

# Path to your best trained model
final_model_path = 'models/MobileNetV2/mobilenet_model_ft.keras' # Or your desired model path

# Load the model
try:
    final_model = tf.keras.models.load_model(final_model_path)
    print(f"Successfully loaded model from {final_model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    # Handle the error, e.g., exit or use a fallback
    exit() # Exit if model cannot be loaded

# Make a prediction
predictions = final_model.predict(img_array)

# 'predictions' will be a NumPy array like [[prob_class_0, prob_class_1, ..., prob_class_N-1]]
# because you fed in a batch of 1 image.

print(f"Raw predictions (probabilities): {predictions}")

# Get the predicted class index (the one with the highest probability)
predicted_class_index = np.argmax(predictions[0])

# Get the confidence score for the predicted class
confidence = predictions[0][predicted_class_index]

# Get the class names from your generator (you'll need these to map index to name)
# If you don't have the generator object anymore, you'll need to define your class names manually
# based on the order your generator assigned them.
# Example: class_names = ['class_a', 'class_b', 'class_c', 'class_d']
# A safer way to get class names if your original generator used `flow_from_directory`
# is to re-create a dummy generator or store the class_indices mapping.
# For example, if your training folders were 'Belvedere', 'Hofburg', 'Schonbrunn', 'StStephans'
class_names = ['Karlskirche', 'Riesenrad', 'Schoenbrunn Palace', 'Stephansdom'] # IMPORTANT: Define your class names in the correct order!
# If you have access to your `train_gen` object:
# class_names = list(train_gen.class_indices.keys())
# class_names.sort(key=lambda x: train_gen.class_indices[x]) # To ensure alphabetical order as flow_from_directory usually sorts them

predicted_class_name = class_names[predicted_class_index]

print(f"\nPredicted class: {predicted_class_name}")
print(f"Confidence: {confidence:.4f}")

# Optional: Display the image with its prediction
import matplotlib.pyplot as plt

plt.imshow(img)
plt.title(f"Predicted: {predicted_class_name} ({confidence*100:.2f}%)")
plt.axis('off')
plt.show()