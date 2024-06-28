import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define paths
dataset_path = 'train'  # Assuming train folder contains training images categorized by folders
categories = os.listdir(dataset_path)

# Image size for resizing
IMG_SIZE = 224

# Prepare data
data = []
labels = []

for category in categories:
    category_path = os.path.join(dataset_path, category)
    class_num = categories.index(category)
    for img in os.listdir(category_path):
        try:
            img_array = cv2.imread(os.path.join(category_path, img))
            img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            data.append(img_array)
            labels.append(class_num)
        except Exception as e:
            print(e)

# Convert to numpy arrays and normalize
data = np.array(data) / 255.0
labels = np.array(labels)

# One-hot encode labels
labels = to_categorical(labels, num_classes=len(categories))

# Split data
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(categories), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))

# Save the model
model.save('fruit_veg_model.h5')

# Print the categories used in training
print("Categories:", categories)
