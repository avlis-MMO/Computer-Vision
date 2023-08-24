import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minDetectionCon = 0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)
        
    def findFaces(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxes = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                h,w,c = img.shape
                bboxC = detection.location_data.relative_bounding_box
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                    int(bboxC.width * w), int(bboxC.height * h)
                bboxes.append([bbox, detection.score])
                cv2.rectangle(img,bbox,(255,0,255),2)
                cv2.putText(img, str(int(detection.score[0]*100))+'%', (bbox[0],bbox[1]-20), 
                            cv2.FONT_HERSHEY_PLAIN, 2,(0,0,0), 2)
        return img,bboxes

def main():
    cap = cv2.VideoCapture("C:\\Users\\gonca\\Desktop\\TheGoodStuff\\Courses\\Adavced CV\\videos\\6.mp4")
    ctime = 0
    ptime = 0
    detector = FaceDetector()
    while True:
        sucess, img = cap.read()
        h,w,_ = img.shape
        img = cv2.resize(img, (int(0.2*w),int(0.2*h)))
        img,_ = detector.findFaces(img)
        

        ctime = time.time()
        fps=1/(ctime-ptime)
        ptime = ctime
        cv2.putText(img, str(int(fps)), (30,50), cv2.FONT_HERSHEY_PLAIN, 2,(0,0,0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)




if  __name__ == "__main__":

    main()