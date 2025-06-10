import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam # for M1/M2 chips
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from explore_data import base_path
import matplotlib.pyplot as plt

# Image settings
img_size = (224, 224)  # MobileNetV2 için önerilen boyut
batch_size = 8
epochs = 10
num_classes = 4

# Create data generators
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=.2,
    brightness_range = [0.3, 1.5],
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest' 
)

train_gen = datagen.flow_from_directory(
    base_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    base_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# --- Phase 1: Feature Extraction (Frozen Base) ---

# Build the model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# Compile with higher learning rate
model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

# Set callbacks (optional)
# callbacks = [
#     EarlyStopping(
#         monitor='val_loss', # or 'val_accuracy'
#         patience=3,        # How many epochs to wait after last improvement
#         restore_best_weights=True
#     ),
#     ModelCheckpoint(
#         'models/MobileNetV2/best_model.keras', # Path to save the best model
#         monitor='val_loss',
#         save_best_only=True,
#         mode='min', # Save when val_loss is minimum
#         verbose=1
#     )
# ]

# Train
history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)


# --- Phase 2: Fine-Tuning (Still Frozen Base, Lower LR) ---

# Fine-Tune
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Set callbacks (optional)
# callbacks_ft = [
#     EarlyStopping(
#         monitor='val_loss', # or 'val_accuracy'
#         patience=3,        # How many epochs to wait after last improvement
#         restore_best_weights=True
#     ),
#     ModelCheckpoint(
#         'models/MobileNetV2/best_model_ft.keras', # Path to save the best model
#         monitor='val_loss',
#         save_best_only=True,
#         mode='min', # Save when val_loss is minimum
#         verbose=1
#     )
# ]

history_ft = model.fit(train_gen, validation_data=val_gen, epochs=epochs)

# Save final model (FT version)
os.makedirs("models", exist_ok=True)
model.save('models/MobileNetV2/mobilenet_model_ft.keras')

# Plot accuracy (phase 1 and 2)
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history_ft.history['accuracy'], label='Train Accuracy (Fine-Tuning)')
plt.plot(history_ft.history['val_accuracy'], label='Validation Accuracy (Fine-Tuning)')
plt.title('MobileNetV2 Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("models/MobileNetV2/mobilenet_accuracy.png")
plt.show()

# Plot loss (phase 1 and 2)
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history_ft.history['loss'], label='Train Loss (Fine-Tuning)')
plt.plot(history_ft.history['val_loss'], label='Validation Loss (Fine-Tuning)')
plt.title('MobileNetV2 Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig("models/MobileNetV2/mobilenet_loss.png")
plt.show()