import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from sort import *

# Initialize the video capture using the specified video file
cap = cv2.VideoCapture("./CarCounter/Videos/cars.mp4")

# Load the YOLO model from the specified weights file
model = YOLO("./Yolo-Weights/yolov8n.pt")

# List of class names corresponding to YOLO's output classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Load a mask image
mask = cv2.imread("./CarCounter/mask.png")

# Initialize the SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Line coordinates for counting
line = [144, 145, 388, 149]

# List to store detected and tracked object IDs
totalCount = []

while True:
    new_frame_time = time.time()

    # Read a frame from the video capture
    success, img = cap.read()
    img = cv2.resize(img, (640, 380))

    # Apply the mask to the image
    imgReg = cv2.bitwise_and(img, mask)

    # Run object detection using the YOLO model
    results = model(imgReg, stream=True)

    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract bounding box coordinates and other information
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentclass = classNames[cls]

            if currentclass in ["car", "bus", "truck", "motorbike"] and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Update the tracker with detections
    resultsTracker = tracker.update(detections)

    # Draw counting line
    cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 3)

    # Iterate through tracked objects
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        # Draw tracked bounding box and ID
        cvzone.cornerRect(img, (x1, y1, w, h), l=3, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, str(id), (max(0, x1), max(35, y1)), scale=0.75, thickness=1, offset=3)

        cx, cy = x1 + w / 2, y1 + h / 2

        # Check if object crosses the counting line
        if line[0] < cx < line[2] and line[1] - 10 < cy < line[3] + 10:
            if id not in totalCount:
                totalCount.append(id)
            print(totalCount)

    # Display the total count
    cvzone.putTextRect(img, "Count: " + str(len(totalCount)), (50, 50), scale=1.5)

    # Display the processed image
    cv2.imshow("Image", img)
    cv2.waitKey(1)

