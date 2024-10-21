FROM python:3.9-slim

# Install dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Pillow explicitly if not in requirements.txt
RUN pip install --no-cache-dir Pillow

# Copy app files
COPY . .

# Expose port
EXPOSE 5001

# Command to run the application
CMD ["python", "app.py"]
