import cv2
import numpy as np
import os
import pyautogui as pg
import handtrackmod as htm  # Costume module

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

# Set parameters for the webcam and movement smoothing
wCam, hCam = 640, 480
frameR = 100
smooth = 2

# Set the webcam resolution
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize variables to track time
ctime = 0
ptime = 0

# Initialize variables to store previous and current mouse locations
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Initialize the hand detection object using the custom handtrackmod
detector = htm.handDetector()

# Get the screen size using pyautogui
wScr, hScr = pg.size()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip the image horizontally for better interaction

    # Find hand landmarks
    img = detector.findHands(img)

    # Get landmarks
    lmlist = detector.findPosition(img, draw=False)
    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]

        # Check fingers up
        fingers = detector.fingersUp()

        # Moving mode
        if fingers[1] and not fingers[2]:
            # Convert coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # Smooth values for mouse movement
            clocX = plocX + (x3 - plocX) / smooth
            clocY = plocY + (y3 - plocY) / smooth

            # Move the mouse cursor
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            pg.moveTo(clocX, clocY)
            plocX, plocY = clocX, clocY

        # Clicking mode
        if fingers[1] and fingers[2] and not fingers[3]:
            # Calculate the distance between two landmarks
            length, img, points = detector.findDist(8, 12, img)
            if length < 50:
                pg.click()
                cv2.circle(img, (points[4], points[5]), 10, (0, 255, 0), cv2.FILLED)

    # Calculate and display the frame rate
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)), (30, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    # Display the processed image
    cv2.imshow("img", img)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()



