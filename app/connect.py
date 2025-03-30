from flask import Flask, request, jsonify
from PIL import Image
import time
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

MODEL_DIR = "../models"
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "garbage_classifier.tflite")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
DISPOSAL_MAP_PATH = os.path.join(MODEL_DIR, "disposal_mapping.json")
IMG_SIZE = (224, 224)


class_names = None
disposal_mapping = None
interpreter = None


def load_model_and_classes():
    global class_names, disposal_mapping, interpreter
    
    try:
        print("Loading TFLite model and class information...")
        
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()
        
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
        
        with open(DISPOSAL_MAP_PATH, 'r') as f:
            disposal_mapping = json.load(f)
            
        print(f"TFLite model loaded successfully. Found {len(class_names)} classes.")
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
        
        # Simulate classification processing (replace this with actual ML model code)
        time.sleep(5)  # Simulate processing delay
        classification_result = classify_image(image)  # Replace with your classification logic
        
        # Return the classification result as JSON
        return jsonify({"classification": classification_result})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def classify_image(image):
    """Classify the uploaded image using the TFLite model"""
    global interpreter, class_names, disposal_mapping
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
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
    
    # Add batch dimension and ensure correct data type
    input_data = np.expand_dims(preprocessed, axis=0).astype(input_details[0]['dtype'])
    
    # Set tensor data
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get output predictions
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    
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



@app.route('/health', methods=['GET'])
def health_check():
    if interpreter is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 503
    return jsonify({"status": "ok", "message": "Service is running"}), 200

if __name__ == '__main__':
    # Load model before starting the server
    if load_model_and_classes():
        app.run(debug=True)
    else:
        print("Failed to load model. Exiting.")

