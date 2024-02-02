# Use an official Python runtime as a parent image
FROM docker.phonepe.com/pp-focal-python-3.5

# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment and activate it
RUN python -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Make port 5003 available to the world outside this container
EXPOSE 5003

# Define environment variable for the model path
ENV MODEL_PATH=yolo-Weights/yolov8n.pt

# Run app.py when the container launches
CMD ["python3", "./videoData.py"]

