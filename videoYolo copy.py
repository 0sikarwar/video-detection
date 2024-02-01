import json
from ultralytics import YOLO
import cv2
import math
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

# Specify the path to your video file
video_path = "./IMG_2466.MOV"

# Model
model = YOLO("yolo-Weights/yolov8s.pt")

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

# Start video capture from file
cap = cv2.VideoCapture(video_path)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Variables to store information about the frame with max persons
max_persons_frame = None
max_persons_count = 0

# Create a list to store object counts for each timestamp
timestamp_object_counts = []

frame_count = 0
frame_interval = 1  # Process every frame

while True:
    success, img = cap.read()

    if not success:
        print("End of video")
        break

    frame_count += 1

    # Process every frame
    if frame_count % frame_interval == 0:
        # Retrieve timestamp for the current frame
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        results = model(img, stream=True)

        # Create a dictionary to store object counts for the current timestamp
        current_timestamp_object_counts = {}

        # Count of persons in the current frame
        persons_count = 0

        person_boxes = []  # Store bounding boxes of detected persons
        chair_boxes = []  # Store bounding boxes of detected chairs
        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

                cls = int(box.cls[0])
                class_name = classNames[cls]

                # Update object count for the current timestamp
                current_timestamp_object_counts[class_name] = current_timestamp_object_counts.get(class_name, 0) + 1

                if class_name == "person":
                    persons_count += 1
                    person_boxes.append([x1, y1, x2, y2])
                elif class_name == "chair":
                    chair_boxes.append([x1, y1, x2, y2])
                    filled = is_chair_filled([x1, y1, x2, y2], person_boxes)
                    color = (0, 255, 0) if filled else (0, 0, 255)  # Green for filled, Red for empty
                    label = "Filled Chair" if filled else "Empty Chair"
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        print(f"Timestamp: {timestamp:.2f} seconds, Persons Count: {persons_count}")

        # Update max persons frame information if needed
        if persons_count > max_persons_count:
            max_persons_count = persons_count
            max_persons_frame = img.copy()

        # Append the object counts for the current timestamp to the list
        timestamp_object_counts.append({"timestamp": timestamp, "object_counts": current_timestamp_object_counts.copy(), "persons_count": persons_count})

        cv2.imshow('Video Playback', img)

    if cv2.waitKey(1) == ord('q'):
        break

# Save object counts for each timestamp to a JSON file
with open('timestamp_object_counts.json', 'w') as json_file:
    json.dump(timestamp_object_counts, json_file, indent=4)

# Save the frame with max persons
if max_persons_frame is not None:
    cv2.imwrite('frame_with_max_persons.jpg', max_persons_frame)

# Display the frame with max persons and count
if max_persons_frame is not None:
    cv2.putText(max_persons_frame, f"Max Persons Count: {max_persons_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Frame with Max Persons', max_persons_frame)
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
