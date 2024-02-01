import json
import time
from ultralytics import YOLO
from imutils.video import VideoStream
import cv2

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

frame_count = 0
frame_interval = 25  # Set to 25 for approximately 500ms interval (25 frames per second)

while True:
    img = vs.read()

    frame_count += 1

    # Process every 500 milliseconds (25 frames * 20 milliseconds per frame)
    if frame_count % frame_interval == 0:
        results = model(img, stream=True)

        # Filter out chairs from the detection results
        chair_results = [box for result in results for box in result.boxes if classNames[int(box.cls[0])] == "chair"]

        # Draw bounding boxes only for empty chairs
        for box in chair_results:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            color = (0, 255, 0)  # Green color for bounding boxes
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, "Empty Chair", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the image using cv2.imshow
        cv2.imshow("Detection Output", img)
        cv2.waitKey(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video stream and close all windows
vs.stop()
cv2.destroyAllWindows()
