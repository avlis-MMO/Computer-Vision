import cv2
import handtrackmod as htm  # Costume module handtrackmod
import time
import numpy as np
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Set the webcam resolution
wCam, hCam = 640, 480

# Initialize the webcam capture
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize variables to track time
ptime = 0

# Initialize the hand detection object using the custom handtrackmod
detector = htm.handDetector(detectionCon=0.75)

# Get the audio device and interface
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Get the volume range
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

# Initialize variables to control volume
vol = 0
volBar = 400
points = []

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)
    if len(lmlist) != 0:

        # Calculate distance between thumb tip and index finger tip
        length, img, points = detector.findDist(4, 8, img, r=10, t=2)

        # Map the hand distance to volume range
        vol = np.interp(length, [50, 150], [minVol, maxVol])
        volBar = np.interp(length, [50, 150], [400, 150])
        per = np.interp(length, [50, 150], [0, 100])

        # Set the volume and display the percentage
        volume.SetMasterVolumeLevel(vol, None)
        cv2.putText(img, "Vol " + str(int(per)) + " %", (30, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

        # Visual feedback when hand is close
        if length < 50:
            cv2.circle(img, (int(points[4]), int(points[5])), 10, (0, 255, 0), cv2.FILLED)

        # Draw volume control range and bar
        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), 3, cv2.FILLED)

    else:
        # If no hand is detected, show current volume percentage
        vol = volume.GetMasterVolumeLevel()
        per = np.interp(vol, [-63, 0], [0, 100])
        cv2.putText(img, "Vol " + str(int(per)) + " %", (30, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

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