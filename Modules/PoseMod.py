# Import necessary libraries
import cv2
import mediapipe as mp
import time
import math


class poseDetector():
    def __init__(self, mode=False, upper_only=False, model_complex=1, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
        # Initialize the poseDetector class with provided parameters
        self.mode = mode
        self.upper_only = upper_only
        self.model_complexity = model_complex
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Initialize mediapipe modules for pose detection
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upper_only, self.model_complexity, self.smooth,
                                     self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        # Process the image using the poseDetector model
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                # Draw landmarks and connections on the image
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)

        return img

    def getPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    # Draw circles at landmark positions
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate angle between landmarks 3 and 1 with landamark 2 as center
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle = angle + 360

        if draw:
            # Draw lines, circles, and text to visualize the angle
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.line(img, (x3, y3), (x2, y2), (255, 0, 0), 2)
            cv2.circle(img, (x1, y1), 6, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 6, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 6, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (int((x1 + x3) / 2), int((y1 + y3) / 2)),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

        return angle


def main():
    cap = cv2.VideoCapture('C:\\Users\\gonca\\Desktop\\TheGoodStuff\\Courses\\Adavced CV\\videos\\4.mp4')
    pTime = 0
    cTime = 0
    detector = poseDetector()  # Create an instance of poseDetector
    while True:
        success, img = cap.read()  # Read a frame from the video
        h, w, c = img.shape
        img = cv2.resize(img, (int(w * 0.2), int(h * 0.2)))
        img = detector.findPose(img)  # Find and draw pose landmarks
        lmList = detector.getPosition(img)  # Find the positions of landmarks

        cTime = time.time()
        fps = 1 / (cTime - pTime)  # Calculate frames per second
        pTime = cTime

        cv2.putText(img, str(int(fps)), (30, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        cv2.imshow("Image", img)  # Display the processed frame
        cv2.waitKey(1)


if __name__ == "__main__":
    main()