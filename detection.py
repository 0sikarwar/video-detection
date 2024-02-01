import json
import time
from ultralytics import YOLO
from imutils.video import VideoStream
import cv2
import numpy as np

# Function to check if a chair is filled
def is_chair_filled(chair_box, person_boxes, threshold=50):
    """
    Determines if a chair is filled based on the proximity of any person to the chair.
    :param chair_box: Bounding box of the chair (x1, y1, x2, y2).
    :param person_boxes: List of bounding boxes of detected persons.
    :param threshold: Distance threshold to consider a chair filled.
    :return: True if filled, False otherwise.
    """
    chair_center = np.array([(chair_box[0] + chair_box[2]) / 2, (chair_box[1] + chair_box[3]) / 2])
    for person_box in person_boxes:
        person_center = np.array([(person_box[0] + person_box[2]) / 2, (person_box[1] + person_box[3]) / 2])
        distance = np.linalg.norm(chair_center - person_center)
        if distance < threshold:
            return True
    return False

# Model
model = YOLO("yolo-Weights/yolov8n.pt")

# Object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Open the live streaming URL
url = "http://127.0.0.1:5001/video_feed"  # Replace with your actual video stream URL
vs = VideoStream(src=url).start()

frame_count = 0
frame_interval = 25  # Process every 25th frame for real-time performance

while True:
    img = vs.read()
    frame_count += 1

    if frame_count % frame_interval == 0:
        results = model(img, stream=True)

        person_boxes = []  # Store bounding boxes of detected persons
        chair_boxes = []  # Store bounding boxes of detected chairs

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values
                
                cls = int(box.cls[0])
                if classNames[cls] == "person":
                    person_boxes.append([x1, y1, x2, y2])
                elif classNames[cls] == "chair":
                    chair_boxes.append([x1, y1, x2, y2])

        # Check each chair if it is filled or not, and draw boxes accordingly
        for chair_box in chair_boxes:
            filled = is_chair_filled(chair_box, person_boxes)
            color = (0, 0, 255) if filled else (0, 255, 0)  # Red for filled, Green for empty
            label = "Filled Chair" if filled else "Empty Chair"
            x1, y1, x2, y2 = chair_box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the image
        cv2.imshow("Detection Output", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Clean up
vs.stop()
cv2.destroyAllWindows()
