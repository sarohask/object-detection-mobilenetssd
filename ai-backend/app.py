from flask import Flask, request, jsonify
import cv2
import numpy as np
import json
import os

app = Flask(__name__)

# Load the MobileNet SSD model
prototxt_path = "models/mobilenet_ssd_deploy.prototxt"
model_path = "models/mobilenet_ssd_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Define the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Define output directory
output_dir = "output"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Read the image
    npimg = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image format"}), 400

    # Prepare the image for MobileNet SSD
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # Pass the blob through the network and get detections
    net.setInput(blob)
    detections = net.forward()

    # List to store detections
    detection_results = []

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Only consider strong detections
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            detection_results.append({
                "label": label,
                "confidence": float(confidence),
                "box": [int(startX), int(startY), int(endX), int(endY)]
            })

    # Save the JSON output
    json_output_path = os.path.join(output_dir, "detections.json")
    with open(json_output_path, 'w') as json_file:
        json.dump({"detections": detection_results}, json_file)
    print(f"JSON saved at: {json_output_path}")

    # Draw bounding boxes on the image
    for detection in detection_results:
        (startX, startY, endX, endY) = detection["box"]
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Save the image with bounding boxes
    output_image_path = os.path.join(output_dir, "bounding_box_image.jpg")
    cv2.imwrite(output_image_path, image)
    print(f"Image saved at: {output_image_path}")


    # Check if files were saved successfully
    if os.path.exists(json_output_path) and os.path.exists(output_image_path):
        return jsonify({
            "message": "Files saved successfully.",
            "detections": detection_results,
            "json_file": json_output_path,
            "image_file": output_image_path
        })
    else:
        return jsonify({"error": "Failed to save files."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
