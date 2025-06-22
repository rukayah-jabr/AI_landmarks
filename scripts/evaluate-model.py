import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Path to your best trained model
model_suffix = "v5_adv_ft2"
model_path = f'models/MobileNetV2/best_model_{model_suffix}.keras' # Or your desired model path

# Data path
val_path = os.path.abspath(os.path.join("data", "landmarks_new"))
test_path = os.path.abspath(os.path.join("data", "test"))

all_classes = ['DC_Tower', 'FH_Campus_Wien', 'Hundertwasserhaus', 'Karlskirche', 'Millenium_Tower', 'Schloss_Belvedere', 'Schloss_Schoenbrunn', 'Secession', 'Stephansdom', 'Wiener_Hofburg', 'Wiener_Riesenrad', 'Wiener_Staatsoper']

# Load the model
try:
    model = tf.keras.models.load_model(model_path)
    print(f"Successfully loaded model from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    # Handle the error, e.g., exit or use a fallback

model.summary()

# --- Confusion matrix ---

# Image settings
img_size = (224, 224)
batch_size = 32

val_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
).flow_from_directory(
    val_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

test_gen = ImageDataGenerator(
    rescale=1./255
).flow_from_directory(
    test_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    classes=all_classes
)

# Predict class probabilities
y_pred = model.predict(val_gen)

# Get predicted class labels
y_pred_labels = np.argmax(y_pred, axis=1)

# Get true class labels from generator
y_true = val_gen.classes

# Get class names (optional)
class_names = list(val_gen.class_indices.keys())
print(class_names)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred_labels)

# Plot using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# --- Evaluation metrics (validation set) ---

print(classification_report(y_true, y_pred_labels, target_names=class_names))