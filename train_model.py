import os
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt

TRAIN_DIR = 'emotion_dataset/train' # Path to training images
TEST_DIR = 'emotion_dataset/test'   # Path to testing images

IMG_height = 48
IMG_width = 48
BATCH_SIZE = 64  # Number of images to process at once

# Data Augmentation
print("Loading Data...")

train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalize pixel values to 0-1
    rotation_range=15,        # Randomly rotate images
    width_shift_range=0.1,    # Randomly shift images horizontally
    height_shift_range=0.1,   # Randomly shift images vertically
    shear_range=0.1,          # Randomly shear images
    zoom_range=0.1,           # Randomly zoom in
    horizontal_flip=True,     # Randomly flip images horizontally
    fill_mode='nearest'       # How to fill newly created pixels
)


# For testing, we only rescale the images (no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_height, IMG_width),
    batch_size=BATCH_SIZE,
    class_mode='categorical', # Used for multi-class classification
    color_mode='grayscale',   # Emotion detection works well on grayscale images
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_height, IMG_width),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=False
)

print(f"Classes found: {train_generator.class_indices}")

# Model Architecture

model = Sequential()

# Block 1
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(IMG_height, IMG_width, 1)))
model.add(BatchNormalization()) # Stabilizes training
model.add(MaxPooling2D(pool_size=(2, 2))) # Reduces image size
model.add(Dropout(0.25)) # Prevents overfitting by randomly turning off neurons

# Block 2
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Block 3
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Block 4
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flatten & Dense Layers
model.add(Flatten()) # Converts 2D feature maps to 1D vector
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Output Layer
# 7 neurons because we have 7 emotion classes (angry, disgust, fear, happy, neutral, sad, surprise)
model.add(Dense(7, activation='softmax'))

# Compile the model
# Adam is an efficient optimizer. Categorical Crossentropy is standard for multi-class classification.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Training the model

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# Add ModelCheckpoint to save the best model during training
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint(
    filepath='face Emotion detection model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Compute Class Weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))
print(f"Class Weights: {class_weights_dict}")

print("Starting Training...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=30, # Optimized epochs for better convergence. Can stop anytime as it saves checkpoints.
    validation_data=test_generator,
    validation_steps=test_generator.samples // BATCH_SIZE,
    class_weight=class_weights_dict,
    callbacks=[reduce_lr, early_stopping, checkpoint]
)

model_name = 'face Emotion detection model.h5'
model.save(model_name)
print(f"Model saved as {model_name}")
