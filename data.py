from flask import Flask, render_template, Response
import json
import time
from ultralytics import YOLO
from imutils.video import VideoStream
import cv2
import math

app = Flask(__name__)

# Model
model = YOLO("yolo-Weights/yolov8n.pt")

# Object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", " pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Open the live streaming URL
url = "http://127.0.0.1:5001/video_feed"  # Replace with your actual video stream URL
vs = VideoStream(src=url).start()

# Create a list to store object counts for each timestamp
timestamp_object_counts = []

frame_count = 0
frame_interval = 25  # Set to 25 for approximately 500ms interval (25 frames per second)

@app.route('/')
def index():
    return render_template('data.html')

def generate():
    global frame_count

    while True:
        img = vs.read()

        frame_count += 1

        # Process every 500 milliseconds (25 frames * 20 milliseconds per frame)
        if frame_count % frame_interval == 0:
            # Retrieve timestamp for the current frame
            timestamp = time.time()

            results = model(img, stream=True)

            # Create a dictionary to store object counts for the current timestamp
            current_timestamp_object_counts = {}

            # Coordinates
            for r in results:
                boxes = r.boxes

                for box in boxes:
                    # Bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

                    # Class name
                    cls = int(box.cls[0])

                    # Update object count for the current timestamp
                    current_timestamp_object_counts[classNames[cls]] = current_timestamp_object_counts.get(classNames[cls], 0) + 1

            # Append the object counts for the current timestamp to the list
            timestamp_object_counts.append({"timestamp": timestamp, "object_counts": current_timestamp_object_counts.copy()})

            # Yield the object counts as JSON
            yield f"data:{json.dumps(current_timestamp_object_counts)}\n\n"

    vs.stop()

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5002, debug=True)

