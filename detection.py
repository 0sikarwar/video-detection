import json
import time
from ultralytics import YOLO
from imutils.video import VideoStream
import cv2
import numpy as np

# Initialize YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")

# Object classes (ensure this matches your model's classes)
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
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

                cls = int(box.cls[0])
                label = classNames[cls]  # Get class name using class index
                confidence = f"{box.conf[0]:.2f}"  # Confidence of detection

                # Draw bounding box and label
                color = (0, 255, 0) if label != "chair" else (255, 0, 0)  # Different color for chairs
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{label} {confidence}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the image
        cv2.imshow("Detection Output", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Clean up
vs.stop()
cv2.destroyAllWindows()
