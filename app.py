from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the model
try:
    model = load_model('fruit_veg_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)

# Categories should match the labels used during training
categories = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']
  # Add all your classes here

# Define image preprocessing function
def preprocess_image(image_path):
    IMG_SIZE = 224
    img_array = cv2.imread(image_path)
    if img_array is None:
        raise ValueError("Image not found or unable to read")
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# Endpoint for the root URL
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint for image upload and prediction
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Preprocess the image
        try:
            processed_image = preprocess_image(file_path)
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        
        # Predict the ingredient
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)
        ingredient = categories[predicted_class]
        
        # Recommend recipes based on the ingredient
        recipes = recommend_recipes(ingredient)
        
        return jsonify({'ingredient': ingredient, 'recipes': recipes})

def recommend_recipes(ingredient):
    recipe_db = {
        "Apple": ["Apple Pie", "Apple Crumble", "Apple Salad"],
        "Banana": ["Banana Bread", "Banana Smoothie", "Banana Pancakes"],
        "Carrot": ["Carrot Soup", "Carrot Cake", "Carrot Salad"],
        "Orange": ["Orange Juice", "Orange Cake", "Orange Salad"],
        "Potato": ["Mashed Potatoes", "Potato Salad", "Potato Soup"]
    }
    return recipe_db.get(ingredient, ["No recipes found for this ingredient"])

if __name__ == '__main__':
    app.run(debug=True)
