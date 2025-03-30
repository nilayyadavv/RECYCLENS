from flask import Flask, request, jsonify
from PIL import Image
import time
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Global variables for model and class information
MODEL_DIR = "../models"
MODEL_PATH = os.path.join(MODEL_DIR, "garbage_classifier.h5")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
DISPOSAL_MAP_PATH = os.path.join(MODEL_DIR, "disposal_mapping.json")
IMG_SIZE = (224, 224)

# Load model and class information on startup
model = None
class_names = None
disposal_mapping = None

def load_model_and_classes():
    global model, class_names, disposal_mapping
    
    try:
        print("Loading classification model and class information...")
        model = load_model(MODEL_PATH, compile=False)
        
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
        
        with open(DISPOSAL_MAP_PATH, 'r') as f:
            disposal_mapping = json.load(f)
            
        print(f"Model loaded successfully. Found {len(class_names)} classes.")
        return True
    except Exception as e:
        print(f"Error loading model or class information: {str(e)}")
        return False

@app.route('/upload', methods=['POST'])
def upload_image():
    # Check if a file is included in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    try:
        # Open the image file for processing
        image = Image.open(file.stream)
        
        # Perform classification
        classification_result = classify_image(image)
        
        # Return the classification result as JSON
        return jsonify(classification_result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def classify_image(image):
    """Classify the uploaded image using the pre-trained model"""
    global model, class_names, disposal_mapping
    
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Resize to model's expected input dimensions
    resized = tf.image.resize(img_array, IMG_SIZE)
    
    # Make sure we have 3 channels (RGB)
    if len(resized.shape) == 2:  # Grayscale
        resized = tf.stack([resized, resized, resized], axis=-1)
    elif resized.shape[-1] == 4:  # RGBA
        resized = resized[:, :, :3]
    
    # Preprocess for EfficientNet
    preprocessed = tf.keras.applications.efficientnet.preprocess_input(resized)
    
    # Add batch dimension
    input_data = np.expand_dims(preprocessed, axis=0)
    
    # Make prediction
    predictions = model.predict(input_data)[0]
    
    # Get top prediction
    top_class_idx = np.argmax(predictions)
    confidence = float(predictions[top_class_idx] * 100)
    
    # Get class name and disposal category
    class_name = class_names[top_class_idx]
    disposal_category = disposal_mapping.get(class_name, "unknown")
    
    # Return results as a dictionary
    result = {
        "class_name": class_name,
        "disposal_category": disposal_category,
        "confidence": confidence
    }
    
    return result

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 503
    return jsonify({"status": "ok", "message": "Service is running"}), 200

if __name__ == '__main__':
    # Load model before starting the server
    if load_model_and_classes():
        app.run(host="0.0.0.0", port=5050)
    else:
        print("Failed to load model. Exiting.")