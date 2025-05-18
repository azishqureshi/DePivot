# Azish Qureshi
# May 2025
# DePivot

# As of Sunday, May 18, 2025 the training dataset has been fully completed, consisting
# of over 7800 images. The validation dataset has not yet been completed.

# The training code is still subject to noticable change, as it is still in development

# Due to the dataset being too large, the training and validation folders will be empty.
# A Google Drive containing the full dataset will soon be available
# The .h5 file is also not included

import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

train_dir = './signLanguageRecognition/dataset/train'
validation_dir = './signLanguageRecognition/dataset/validation'
model_save_path = './sign_language_model.h5'

# Image and training parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
NUM_CLASSES = 6

# Augments data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=['A', 'B', 'C', 'D', 'E', 'F']
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=['A', 'B', 'C', 'D', 'E', 'F']
)

print(f"Classes: {train_generator.class_indices}")

# Builds the model using MobileNetV2
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = True

# This was done to preserve the base features
for layer in base_model.layers[:100]:
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(1024, activation='relu'),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train
model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stop]
)

model.save(model_save_path)
print(f"Training complete! Model saved to {model_save_path}")