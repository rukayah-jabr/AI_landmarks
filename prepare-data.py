from tensorflow.keras.preprocessing.image import ImageDataGenerator
from explore_data import base_path

# this should do the image preprocessing and augmentation
# todo: add more augmentations: brightness, shear

img_size = (299, 299)
batch_size = 16


datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
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

# ✅ Clearer output
print(f"\n✅ Training set: {train_gen.samples} images, {train_gen.num_classes} classes.")
print(f"✅ Validation set: {val_gen.samples} images, {val_gen.num_classes} classes.")
print("📦 Class labels:", train_gen.class_indices)
