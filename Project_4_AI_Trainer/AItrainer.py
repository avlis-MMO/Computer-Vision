# Import necessary libraries
import cv2
import PoseMod as pm  # Custom module for pose detection
import time
import numpy as np
import math

# Open a video capture object using the specified video file
cap = cv2.VideoCapture('./aitrainer/ait2.mp4')

# Initialize variables to track time and pose
ptime = 0
ctime = 0
detector = pm.poseDetector()  # Initialize the pose detector from your custom module
count = 0
dir = 0

# Infinite loop to process each frame of the video
while True:
    # Read a frame from the video capture
    success, img = cap.read()

    # Resize the frame to a smaller size
    h, w, c = img.shape
    img = cv2.resize(img, (int(w * 0.2), int(h * 0.2)))

    # Use the pose detector to find and annotate pose landmarks on the frame
    img = detector.findPose(img, False)
    lmList = detector.getPosition(img, False)

    # Calculate the angle between specific landmarks to determine pose information
    if len(lmList) != 0:
        # Right arm
        # angle=detector.findAngle(img,12, 14, 16)

        # Left arm
        # angle=detector.findAngle(img,11, 13, 15)

        # Right Leg
        angle = detector.findAngle(img, 28, 26, 24)

        # Left leg
        # angle=detector.findAngle(img,27, 25, 23)

    # For curl
    # per = np.interp(angle,(30,100),(0,100))

    # For squat
    per = np.interp(angle, (90, 160), (0, 100))

    # Check for specific pose patterns and update the count accordingly
    if per == 100:
        if dir == 0:
            count += 0.5
            dir = 1
    if per == 0:
        if dir == 1:
            count += 0.5
            dir = 0

    # Annotate the frame with pose information and counts
    cv2.putText(img, str(int(count)), (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(img, str(int(fps)), (300, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    # Display the annotated frame
    cv2.imshow("Image", img)

    # Wait for a key press event (1ms delay)
    cv2.waitKey(1)
