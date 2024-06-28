import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Define paths
dataset_path = 'path_to_extracted_dataset/Fruit-and-Vegetable Image Recognition/train'
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
