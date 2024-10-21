# Object Detection Microservice Project

## Overview
This project is a microservice-based solution for object detection using mobilenetSSD. The UI backend allows users to upload an image, and the AI backend processes the image to detect objects. The results are displayed on the UI as an image with bounding boxes and a corresponding JSON response.

## Project Structure

## Prerequisites

1. **Docker**: Ensure Docker is installed on your system.
2. **Docker Compose**: Make sure you have `docker-compose` installed.
3. **Internet Connection**: You will need a minimal amount of internet to pull Docker images and dependencies.

## Setup Instructions

### 1. Clone the Repository
Download or clone the project to your local machine:
```bash
git clone https://github.com/sarohask/object-detection-mobilenetssd.git
cd object-detection-mobilenetssd
docker-compose up --build
```

### 2. Upload Image
The above command will create a UI localhost webpage for uploading an image. You can select the image and upload it.

### 3. Output
The 'output' folder will contain the JSON file and image with bounding boxes.
