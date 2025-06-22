import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image # For image loading utilities
import matplotlib.pyplot as plt

# Define the path to your single image
directory_path = 'data/test' # IMPORTANT: Change this to your image path
image_files = os.listdir(directory_path)
print(image_files)

# Define the target size that your model expects (224x224 for MobileNetV2)
img_size = (224, 224)

images = []
for img in image_files:
    # 1. Load the image
    img = image.load_img(f"{directory_path}/{img}", target_size=img_size)

    # 2. Convert the image to a NumPy array
    img_array = image.img_to_array(img)

    # 3. Rescale the pixel values (same as your ImageDataGenerator's rescale=1./255)
    img_array = img_array / 255.0

    images.append(img_array)

# Convert to batch
images_np = np.array(images)


# Path to your best trained model
final_model_path = 'models/MobileNetV2/best_model_v5_adv_ft_unfreeze2.keras' # Or your desired model path

# Load the model
try:
    final_model = tf.keras.models.load_model(final_model_path)
    print(f"Successfully loaded model from {final_model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    # Handle the error, e.g., exit or use a fallback
    exit() # Exit if model cannot be loaded

# Make a prediction
preds = final_model.predict(images_np)

# If model outputs probabilities for 12 classes
pred_labels = np.argmax(preds, axis=1)
print(pred_labels)

# Optional: map class index to name
class_names = ["DC_Tower", "FH_Campus_Wien", "Hundertwasserhaus", "Karlskirche", "Millenium_Tower", "Schloss_Belvedere", "Schloss_Schoenbrunn", "Secession", "Stephansdom", "Wiener_Hofburg", "Wiener_Riesenrad", "Wiener_Staatsoper"]
idx_to_class = {i: class_name for i, class_name in enumerate(class_names)}

# Print predictions
for i, path in enumerate(image_files):
    print(f"{os.path.basename(path)} ➜ predicted class: {idx_to_class[pred_labels[i]]} ({(preds[i][pred_labels[i]]*100):.4f}%)")


for i in range(len(images_np)):
    plt.imshow(images_np[i])
    plt.title(f"Predicted: {idx_to_class[pred_labels[i]]} ({(preds[i][pred_labels[i]]*100):.4f}%)")
    plt.axis('off')
    plt.show()