# Import necessary libraries
import cv2
import numpy as np
import handtrackmod as htm  # Assuming this is your custom module for hand tracking
import os
import time

# Set camera dimensions
wCam, hCam = 640, 480

# Open a video capture object using the default camera (camera index 0)
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Store the previous time for frame rate calculation
ptime = 0

# Define the folder path where overlay images are stored
folderPath = "C:\\Users\\gonca\\Desktop\\TheGoodStuff\\Courses\\Adavced CV\\imgs"

# List all image files in the folder
mylist = os.listdir(folderPath)
print(mylist)

# Load overlay images and append to overlay list
overlayList = []
for imgPath in mylist:
    img = cv2.imread(f'{folderPath}\\{imgPath}')
    img = cv2.resize(img, (200, 200))
    overlayList.append(img)

# Initialize hand detector
detector = htm.handDetector(detectionCon=0.75)

# Define the finger tip IDs
tipIds = [4, 8, 12, 16, 20]

# Infinite loop to process each frame from the camera
while True:
    # Capture a frame from the camera
    success, img = cap.read()

    # Detect hands in the frame
    detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)

    if len(lmlist) != 0:
        fingers = []

        # Check if fingertips are raised or not
        for id in range(0, 5):
            if lmlist[tipIds[id]][1] < lmlist[tipIds[id] - 2][1]:
                fingers.append(0)
            else:
                fingers.append(1)

        # Count the number of raised fingers
        num = sum(fingers)

        # Display overlay image based on the number of raised fingers
        img[0:200, 0:200] = overlayList[num]

        # Draw a rectangle and display the finger count
        cv2.rectangle(img, (20, 250), (150, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(num), (40, 380), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    # Calculate and display frame rate
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(img, str(int(fps)), (600, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    # Display the image
    cv2.imshow("Img", img)

    # Wait for a key press event (1ms delay)
    cv2.waitKey(1)
