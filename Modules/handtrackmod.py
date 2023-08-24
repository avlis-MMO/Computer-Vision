# Import necessary libraries
import cv2
import mediapipe as mp
import time
import math


class handDetector():
    def __init__(self, mode=False, maxHands=2, model_complex=1, detectionCon=0.5, trackCon=0.5):
        # Initialize the handDetector class with provided parameters
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complex
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Initialize mediapipe modules for hand detection
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]  # Landmark IDs of fingertips

    def findHands(self, img, draw=True):
        # Process the image using the handDetector model
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # Draw landmarks and connections on the image
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    # Draw circles at landmark positions
                    cv2.circle(img, (cx, cy), 20, (255, 0, 255), cv2.FILLED)

        return self.lmList

    def fingersUp(self):
        fingers = []
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 2][1]:
            fingers.append(0)
        else:
            fingers.append(1)
        # Check fingers by comparing Y coordinates of landmarks
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDist(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            # Draw circles and a line between landmarks
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (int(cx), int(cy)), r, (255, 0, 255), cv2.FILLED)

        # Calculate landmark distance
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)  # Open the default camera (camera index 0)
    detector = handDetector()  # Create an instance of handDetector
    while True:
        success, img = cap.read()  # Read a frame from the camera
        img = detector.findHands(img)  # Find and draw hands in the frame
        lmList = detector.findPosition(img)  # Find the positions of landmarks
        if len(lmList) != 0:
            print(lmList[4])  # Print the position of the tip of the thumb

        cTime = time.time()
        fps = 1 / (cTime - pTime)  # Calculate frames per second
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

        cv2.imshow("Image", img)  # Display the processed frame
        cv2.waitKey(1)  # Wait for a key press


if __name__ == "__main__":
    main()