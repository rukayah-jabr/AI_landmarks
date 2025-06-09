import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from explore_data import base_path
import matplotlib.pyplot as plt

base_path = os.path.abspath(os.path.join("data", "landmarks60"))

# Settings
img_size_224 = (224, 224)
img_size_299 = (299, 299)
batch_size = 16
epochs = 5
num_classes = 4

def data_gen(img_size):

    # Create data generators
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=40,
        zoom_range=0.3,
        brightness_range=(0.8, 1.2),
        horizontal_flip=True,
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

    return train_gen, val_gen

def custom_model():
    # Build the custom model
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),  # Add dropout
        layers.Dense(num_classes, activation='softmax')  # 4 classes
    ])
    return model

def inception_model():
    # Build the model
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    base_model.trainable = False  # Freeze the pretrained layers

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')  # 4 classes
    ])
    return model

def mobilenet_model():
    # Build the model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze base

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax') # 4 classes
    ])
    return model

def compile_train_model(model, name, img_size):
    train_gen, val_gen = data_gen(img_size)

    # Compile
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train
    results = model.fit(train_gen, validation_data=val_gen, epochs=epochs)

    # Save model
    os.makedirs("models", exist_ok=True)
    model.save(f'models/{name}_model.keras')

    # Return data for plotting
    return results

def plot_metrics(trained_model, title):
    os.makedirs(f"models/{title}", exist_ok=True)

    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(trained_model.history['accuracy'], label='Train Accuracy')
    plt.plot(trained_model.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{title} Model Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"models/{title}/{title}_accuracy.png")
    # plt.show()

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(trained_model.history['loss'], label='Train Loss')
    plt.plot(trained_model.history['val_loss'], label='Validation Loss')
    plt.title(f'{title} Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"models/{title}/{title}_loss.png")
    # plt.show()


# Custom Model
custom = custom_model()
trained_custom = compile_train_model(custom, "custom", img_size_224)
plot_metrics(trained_custom, "Custom")

# InceptionV3 Model
inception = inception_model()
trained_inception = compile_train_model(inception, "inception", img_size_299)
plot_metrics(trained_inception, "InceptionV3")

# MobileNetV2 Model
mobilenet = mobilenet_model()
trained_mobilenet= compile_train_model(mobilenet, "mobilenet", img_size_224)
plot_metrics(trained_mobilenet, "MobileNetV2")