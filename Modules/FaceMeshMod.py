import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=2, rland=False, minDetecCon=0.5, minTrackCon=0.5):
        # Initialize the FaceMeshDetector class with provided parameters
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.rland = rland
        self.minDetecCon = minDetecCon
        self.minTrackCon = minTrackCon

        # Initialize mediapipe modules
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.rland,
                                                 self.minDetecCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(color=[0, 255, 0], thickness=1, circle_radius=1)

    def FindFaceMesh(self, img, draw=True):
        # Process the image using the FaceMesh model
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLm in self.results.multi_face_landmarks:
                if draw:
                    # Draw landmarks on the image if required
                    self.mpDraw.draw_landmarks(img, faceLm, self.mpFaceMesh.FACEMESH_CONTOURS,
                                               self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLm.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(w * lm.x), int(h * lm.y)
                    face.append([cx, cy])
                faces.append(face)
        return img, faces

def main():
    # Open a video capture object using a video file path
    cap = cv2.VideoCapture("C:\\Users\\gonca\\Desktop\\TheGoodStuff\\Courses\\Adavced CV\\videos\\5.mp4")
    ctime = 0
    ptime = 0
    detector = FaceMeshDetector()
    while True:
        # Capture a frame from the video
        success, img = cap.read()
        h, w, _ = img.shape
        img = cv2.resize(img, (int(0.2 * w), int(0.2 * h)))
        # Use the detector to find face meshes in the image
        img, faces = detector.FindFaceMesh(img)
        if len(faces) != 0:
            print(len(faces))
        # Calculate and display frame rate
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(img, str(int(fps)), (30, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        # Display the processed image
        cv2.imshow("Image", img)
        # Wait for a key press event (1ms delay)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()