import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam # for M1/M2 chips

base_path = os.path.abspath(os.path.join("data", "landmarks_new"))
model_suffix = "v3_adv_ft"

# Settings
img_size = (224, 224)
batch_size = 32
num_classes = 12

# Create data generators
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,          # Smaller rotation
    zoom_range=0.1,             # Less zoom
    brightness_range=[0.8, 1.2],# Subtle brightness change
    width_shift_range=0.05,     # Less shift
    height_shift_range=0.05,
    shear_range=0.05,           # Milder shear
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

# Don't apply augmentation on validation set
val_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
).flow_from_directory(
    base_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Build the custom model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(256, (3,3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),  # Add dropout
    layers.Dense(num_classes, activation='softmax')  # 4 classes
])

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set callbacks (optional)
callbacks = [
    EarlyStopping(
        monitor='val_loss', # or 'val_accuracy'
        patience=7,        # How many epochs to wait after last improvement
        restore_best_weights=True
    ),
    ModelCheckpoint(
        f'models/Custom/best_model_{model_suffix}.keras', # Path to save the best model
        monitor='val_loss',
        save_best_only=True,
        mode='min', # Save when val_loss is minimum
        verbose=1
    )
]

# Train
history = model.fit(train_gen, validation_data=val_gen, epochs=5)

# ---- Phase 2 -----
# Use smaller learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
history_ft = model.fit(train_gen, validation_data=val_gen, epochs=25, callbacks=callbacks)

# Save model
os.makedirs("models", exist_ok=True)
model.save(f'models/Custom/custom_model_{model_suffix}.keras')

# Plot accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history_ft.history['accuracy'], label='Train Accuracy (Fine-Tuning)')
plt.plot(history_ft.history['val_accuracy'], label='Validation Accuracy (Fine-Tuning)')
plt.title(f'Custom Model ({model_suffix}) Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(f"models/Custom/custom_accuracy_{model_suffix}.png")
plt.show()

# Plot loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history_ft.history['loss'], label='Train Loss (Fine-Tuning)')
plt.plot(history_ft.history['val_loss'], label='Validation Loss (Fine-Tuning)')
plt.title(f'Custom Model ({model_suffix}) Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(f"models/Custom/custom_loss_{model_suffix}.png")
plt.show()