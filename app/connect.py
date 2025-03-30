from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    # Save or process the uploaded file
    file.save(f"./uploads/{file.filename}")
    
    return jsonify({"message": "File uploaded successfully!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
