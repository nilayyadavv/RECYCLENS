import os
import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Paths
MODEL_DIR = "../models"
MODEL_PATH = os.path.join(MODEL_DIR, "garbage_classifier.h5")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
DISPOSAL_MAP_PATH = os.path.join(MODEL_DIR, "disposal_mapping.json")
IMG_SIZE = (224, 224)

def load_class_info():
    """Load class names and disposal mapping."""
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
    
    with open(DISPOSAL_MAP_PATH, 'r') as f:
        disposal_mapping = json.load(f)
    
    return class_names, disposal_mapping

def preprocess_image(image):
    """Preprocess image for model input."""
    # Resize to model's expected input dimensions
    resized = cv2.resize(image, IMG_SIZE)
    
    # Convert BGR to RGB (OpenCV loads as BGR, but model expects RGB)
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Preprocess for EfficientNet
    preprocessed = tf.keras.applications.efficientnet.preprocess_input(rgb_image)
    
    # Add batch dimension
    return np.expand_dims(preprocessed, axis=0)

def predict_image(model, image, class_names, disposal_mapping):
    """Predict the class of the image and its disposal category."""
    processed_image = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(processed_image)[0]
    
    # Get top prediction
    top_class_idx = np.argmax(predictions)
    confidence = predictions[top_class_idx] * 100
    
    # Get class name and disposal category
    class_name = class_names[top_class_idx]
    disposal_category = disposal_mapping.get(class_name, "unknown")
    
    return class_name, disposal_category, confidence

def draw_prediction(image, class_name, disposal_category, confidence):
    """Draw prediction info on the image."""
    result_image = image.copy()
    
    # Set up text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    # Add colored background based on disposal category
    color_map = {
        "compost": (0, 255, 0),  # Green for compost
        "recycle": (255, 0, 0),  # Blue for recycle
        "garbage": (0, 0, 255)   # Red for garbage
    }
    
    color = color_map.get(disposal_category, (200, 200, 200))
    
    # Prepare text
    prediction_text = f"Class: {class_name}"
    disposal_text = f"Dispose as: {disposal_category}"
    confidence_text = f"Confidence: {confidence:.1f}%"
    
    # Get text sizes
    (p_w, p_h), _ = cv2.getTextSize(prediction_text, font, font_scale, thickness)
    (d_w, d_h), _ = cv2.getTextSize(disposal_text, font, font_scale, thickness)
    (c_w, c_h), _ = cv2.getTextSize(confidence_text, font, font_scale, thickness)
    
    # Get max width for rectangle
    max_width = max(p_w, d_w, c_w) + 20
    total_height = p_h + d_h + c_h + 40
    
    # Draw rectangle background
    cv2.rectangle(result_image, (10, 10), (10 + max_width, 10 + total_height), 
                 (0, 0, 0), -1)
    
    # Draw prediction texts
    cv2.putText(result_image, prediction_text, (20, 10 + p_h + 5), 
               font, font_scale, color, thickness)
    cv2.putText(result_image, disposal_text, (20, 10 + p_h + d_h + 15), 
               font, font_scale, color, thickness)
    cv2.putText(result_image, confidence_text, (20, 10 + p_h + d_h + c_h + 25), 
               font, font_scale, color, thickness)
    
    # Draw colored disposal indicator
    cv2.rectangle(result_image, (5, 10), (10, 10 + total_height), color, -1)
    
    return result_image

def test_single_image(image_path, model, class_names, disposal_mapping):
    """Test a single image and display results."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Get prediction
    class_name, disposal_category, confidence = predict_image(
        model, image, class_names, disposal_mapping
    )
    
    # Draw prediction on image
    result_image = draw_prediction(image, class_name, disposal_category, confidence)
    
    # Print results
    print(f"Image: {image_path}")
    print(f"Predicted Class: {class_name}")
    print(f"Disposal Category: {disposal_category}")
    print(f"Confidence: {confidence:.1f}%")
    
    # Display image
    cv2.imshow("Garbage Classification Result", result_image)
    cv2.waitKey(0)
    
    return class_name, disposal_category, confidence

def test_from_webcam(model, class_names, disposal_mapping):
    """Test using webcam feed."""
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit, 'c' to capture and classify an image")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Show instructions
        cv2.putText(frame, "Press 'c' to classify, 'q' to quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display frame
        cv2.imshow("Webcam Feed", frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Get prediction
            class_name, disposal_category, confidence = predict_image(
                model, frame, class_names, disposal_mapping
            )
            
            # Draw prediction on image
            result_image = draw_prediction(frame, class_name, disposal_category, confidence)
            
            # Print results
            print(f"Predicted Class: {class_name}")
            print(f"Disposal Category: {disposal_category}")
            print(f"Confidence: {confidence:.1f}%")
            
            # Display result
            cv2.imshow("Classification Result", result_image)
            
            # Wait until a key is pressed to continue with webcam feed
            cv2.waitKey(0)
            cv2.destroyWindow("Classification Result")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return
    
    # Load model
    print("Loading model...")
    model = load_model(MODEL_PATH, compile=False)
    
    # Load class info
    class_names, disposal_mapping = load_class_info()
    print(f"Loaded {len(class_names)} classes: {class_names}")
    
    # Menu
    while True:
        print("\nGarbage Classification Test")
        print("1. Test a single image file")
        print("2. Test using webcam")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            image_path = input("Enter the path to the image file: ")
            test_single_image(image_path, model, class_names, disposal_mapping)
        elif choice == '2':
            test_from_webcam(model, class_names, disposal_mapping)
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")
    
    print("Exiting...")

if __name__ == "__main__":
    main()