from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# Initialize video capture
cap = cv2.VideoCapture("./ppe2.mp4")

# Load YOLO model
model = YOLO("./ppe.pt")

# List of class names for visualization
classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
              'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery',
              'mini-van', 'sedan', 'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    img = cv2.resize(img, (640, 380))
    # Perform object detection using the YOLO model
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # Draw bounding box
            cvzone.cornerRect(img, (x1, y1, w, h), l=5)
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            # Display class name and confidence on the image
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(10, y1)),
                               scale=0.75, thickness=1, offset=5)

    # Calculate and display frames per second
    fps = int(1 / (time.time() - new_frame_time))
    cvzone.putTextRect(img, f'FPS: {fps}', (10, 10), scale=0.5, thickness=1, offset=5)

    # Display the annotated image
    cv2.imshow("Image", img)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()