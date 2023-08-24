# Import necessary libraries
import cv2
import numpy as np
import handtrackmod as htm  # Custom module for hand tracking
import time
import os

# Open a video capture object using the default camera (camera index 0)
cap = cv2.VideoCapture(0)

# Set the camera dimensions
wCam, hCam = 1080, 720
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize brush and eraser thickness
brushThick = 15
eraserThick = 50

# Define the folder path where overlay images are stored
folderPath = "C:\\Users\\gonca\\Desktop\\TheGoodStuff\\Courses\\Adavced CV\\VAI"

# List all image files in the folder
mylist = os.listdir(folderPath)
overlayList = []

# Load overlay images and append to overlay list
for imgPath in mylist:
    image = cv2.imread(folderPath + '\\' + imgPath)
    image = cv2.resize(image, (960, 125))
    overlayList.append(image)

# Set the initial header overlay image and initialize the hand detector
header = overlayList[0]
detector = htm.handDetector(detectionCon=0.75)
drawColor = (30, 30, 255)

# Initialize variables to track previous coordinates
xp, yp = 0, 0

# Create an empty canvas to draw on
imgCanvas = np.zeros((540, 960, 3), np.uint8)

# Infinite loop to process each frame from the camera
while True:
    # Capture a frame from the camera
    success, img = cap.read()

    # Flip the frame horizontally
    img = cv2.flip(img, 1)

    # Detect hands in the frame
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)

    if len(lmlist) != 0:
        # Extract coordinates of specific landmarks
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]

        # Check finger positions
        fingers = detector.fingersUp()

        # Selection mode logic
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
            print("Selection Mode")

            # Update header and draw color based on position
            if y1 < 125:
                if 0 < x1 < 96:
                    header = overlayList[0]
                    drawColor = (30, 30, 255)
                # ... repeat for other positions ...

        # Draw mode logic
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Draw Mode")

            # Track previous point for drawing lines
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            # Draw lines on canvas based on draw color
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThick)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThick)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThick)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThick)

            # Update previous point
            xp, yp = x1, y1

    # Create a negative version of the canvas for erasing
    imgInv = imgCanvas.copy()
    imgInv[imgInv != 0] = 5
    imgInv[imgInv == 0] = 255
    imgInv[imgInv == 5] = 0

    # Combine canvas and image to show the drawing
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Overlay header on top of the image
    img[0:125, 0:960] = header

    # Display the image
    cv2.imshow("image", img)

    # Wait for a key press event (1ms delay)
    cv2.waitKey(1)

