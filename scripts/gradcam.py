import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import cv2
from explore_data import base_path

# Load the trained model
model = load_model('models/landmark_model.h5')

# Load and preprocess a sample image
img_folder = os.path.join(base_path, 'Schoenbrunn Palace Vienna')  # You can change to any test folder
img_file = os.listdir(img_folder)[0]
img_path = os.path.join(img_folder, img_file)

img = image.load_img(img_path, target_size=(299, 299))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Load the InceptionV3 base
inception = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
last_conv_layer = inception.get_layer("mixed10")

# ⚙️ Build a model for Grad-CAM
x = last_conv_layer.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(3, activation='softmax')(x)  # ✅ You have 3 classes
grad_model = tf.keras.models.Model(inputs=inception.input, outputs=[last_conv_layer.output, x])

# 🔬 Use GradientTape to get gradients
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    predicted_class = tf.argmax(predictions[0])
    loss = predictions[:, predicted_class]

print("Predicted class index:", predicted_class.numpy())

# Manual class label mapping (based on your folders)
class_indices = {
    'Schoenbrunn Palace Vienna': 0,
    'Stephansdom Vienna': 1,
    'Wiener Riesenrad Vienna': 2
}
class_labels = {v: k for k, v in class_indices.items()}
print("Predicted class label:", class_labels[predicted_class.numpy()])

# Calculate Grad-CAM heatmap
grads = tape.gradient(loss, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
conv_outputs = conv_outputs[0]
heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)
heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

# Overlay the heatmap on original image
img_orig = cv2.imread(img_path)
img_orig = cv2.resize(img_orig, (299, 299))
heatmap = cv2.resize(heatmap.numpy(), (img_orig.shape[1], img_orig.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(img_orig, 0.6, heatmap_color, 0.4, 0)

# 💾 Save and show result
cv2.imwrite("models/gradcam_result.png", superimposed_img)
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.title(f"Grad-CAM: {class_labels[predicted_class.numpy()]}")
plt.axis('off')
plt.show()
