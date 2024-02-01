from flask import Flask, render_template, Response
import json
import time
from ultralytics import YOLO
from imutils.video import VideoStream
import cv2
import numpy as np

app = Flask(__name__)

# Function to check if a chair is filled
def is_chair_filled(chair_box, person_boxes, threshold=50):
    chair_center = np.array([(chair_box[0] + chair_box[2]) / 2, (chair_box[1] + chair_box[3]) / 2])
    for person_box in person_boxes:
        person_center = np.array([(person_box[0] + person_box[2]) / 2, (person_box[1] + person_box[3]) / 2])
        distance = np.linalg.norm(chair_center - person_center)
        if distance < threshold:
            return True
    return False

# Model initialization
model = YOLO("yolo-Weights/yolov8n.pt")  # Ensure the correct path to model weights

# Open the live streaming URL
url = "http://127.0.0.1:5001/video_feed"  # Update with your actual video stream URL
vs = VideoStream(src=url).start()

@app.route('/')
def index():
    return render_template('data.html')

def generate():
    frame_count = 0
    frame_interval = 25  # Adjust as needed

    while True:
        img = vs.read()
        frame_count += 1

        if frame_count % frame_interval == 0:
            timestamp = time.time()
            results = model(img, stream=True)
            person_boxes = []
            chair_boxes = []

            for r in results:
                boxes = r.boxes

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cls = int(box.cls[0])
                    class_name = model.names[cls]

                    if class_name == "person":
                        person_boxes.append([x1, y1, x2, y2])
                    elif class_name == "chair":
                        chair_boxes.append([x1, y1, x2, y2])

            # Determine empty chairs and total persons
            empty_chairs_count = sum(not is_chair_filled(chair_box, person_boxes) for chair_box in chair_boxes)
            total_persons = len(person_boxes)

            data = {
                "timestamp": timestamp,
                "empty_chairs": empty_chairs_count,
                "total_chairs": len(chair_boxes),
                "total_persons": total_persons
            }

            yield f"data:{json.dumps(data)}\n\n"

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
