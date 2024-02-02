from flask import Flask, render_template, send_file
from flask_socketio import SocketIO
from ultralytics import YOLO
import cv2
import numpy as np
import threading
import time
import json
import io

current_frame = None
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Function to check if a chair is filled
def is_chair_filled(chair_box, person_boxes, threshold=50):
    chair_center = np.array([(chair_box[0] + chair_box[2]) / 2, (chair_box[1] + chair_box[3]) / 2])
    for person_box in person_boxes:
        person_center = np.array([(person_box[0] + person_box[2]) / 2, (person_box[1] + person_box[3]) / 2])
        distance = np.linalg.norm(chair_center - person_center)
        if distance < threshold:
            return True
    return False

# Initialize YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")  # Adjust path to your YOLO model weights

def generate(video_path):
    global current_frame
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        frame_interval = 1  # Adjust as needed for your application's performance
        
        while True:
            success, img = cap.read()
            if not success:
                break  # Exit the loop if no more frames are available

            frame_count += 1
            
            if frame_count % frame_interval == 0:
                timestamp = time.time()
                results = model(img, stream=True)
                person_boxes = []
                chair_boxes = []
                empty_chairs_positions = []

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
                            is_filled = is_chair_filled([x1, y1, x2, y2], person_boxes)
                            if not is_filled:
                                empty_chairs_positions.append([x1, y1, x2, y2])
                            color = (0, 0, 255) if is_filled else (255, 0, 0)  # Red if filled, Blue if empty
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # Determine empty chairs and total persons
                empty_chairs_count = sum(not is_chair_filled(chair_box, person_boxes) for chair_box in chair_boxes)
                total_persons = len(person_boxes)

                data = {
                    "timestamp": timestamp,
                    "empty_chairs": empty_chairs_count,
                    "total_chairs": len(chair_boxes),
                    "total_persons": total_persons,
                    "empty_chairs_positions": empty_chairs_positions.copy()
                }
                print(f"timestamp: {timestamp}")
                current_frame = img.copy()
                socketio.emit('update_data', json.dumps(data))  # Ensure data is sent as a JSON string
        cap.release()
    except Exception as e:
        print(f"Error processing video: {e}")
        socketio.emit('update_data', {'error': str(e)})

@app.route('/')
def index():
    # Start video processing in a separate thread
    threading.Thread(target=generate, args=("./IMG_2469.MOV")).start()
    return render_template('video_data.html')

@app.route('/screenshot')
def screenshot():
    global current_frame
    if current_frame is not None:
        # Convert the current frame to JPEG
        success, encoded_image = cv2.imencode('.jpg', current_frame)
        if not success:
            return "Failed to encode image", 500
        # Convert to bytes and send as a response
        byte_io = io.BytesIO(encoded_image.tobytes())
        return send_file(byte_io, mimetype='image/jpeg')
    else:
        return "No frame available", 404

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5003, debug=True)
