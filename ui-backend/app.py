from flask import Flask, render_template, request, redirect, jsonify
import requests

app = Flask(__name__)

AI_BACKEND_URL = "http://ai-backend:5001/detect"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect('/')
    file = request.files['file']

    if file.filename == '':
        return redirect('/')

    # Send the file to the AI backend
    files = {'file': file.read()}
    response = requests.post(AI_BACKEND_URL, files=files)

    if response.status_code == 200:
        detections = response.json()['detections']
        print("Response from AI backend:", response.json())  # Log the response
        return jsonify(detections)
    else:
        print("Error response from AI backend:", response.text)  # Log error response
        return jsonify({"error": "Detection failed"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
