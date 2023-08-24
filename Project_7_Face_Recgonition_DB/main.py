import os
import numpy as np
import cv2
import cvzone
import pickle
import face_recognition
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime

# Initialize Firebase using the provided credentials file, database URL, and storage bucket
cred = credentials.Certificate("./cred.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://faceattendancerealtime-25805-default-rtdb.firebaseio.com/',
    'storageBucket': 'faceattendancerealtime-25805.appspot.com'
})

bucket = storage.bucket()

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load background image
imgBackGround = cv2.imread("./FaceRecRealTimeDB/Resources/background.png")

# Importing the modes to a list
folderModePath = "./FaceRecRealTimeDB/Resources/Modes"
modePathList = os.listdir(folderModePath)
imgModeList = []

for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

print(len(imgModeList))

# Import the encoding file
file = open("./FaceRecRealTimeDB/EncodeFile.p", "rb")
encodeListKnowWithIDs = pickle.load(file)
file.close()
encodeListKnow, studentsID = encodeListKnowWithIDs
print(studentsID)

modeType = 0
counter = 0
id = -1
imgStudent = []

while True:
    success, img = cap.read()

    # Resize face capture to increase performance
    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Get face location and encoding
    faceCurrFrame = face_recognition.face_locations(imgS)
    encodeCurrFrame = face_recognition.face_encodings(imgS, faceCurrFrame)

    # Prepare the image to be shown
    imgBackGround[162:162 + 480, 55:55 + 640] = img
    imgBackGround[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    if faceCurrFrame:
        for encodFace, faceLoc in zip(encodeCurrFrame, faceCurrFrame):
            mathces = face_recognition.compare_faces(encodeListKnow, encodFace)
            faceDist = face_recognition.face_distance(encodeListKnow, encodFace)

            matchIdx = np.argmin(faceDist)

            # If face found matches ones on database show its information
            if mathces[matchIdx]:
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                bbox = x1 + 55, 162 + y1, x2 - x1, y2 - y1

                imgBackGround = cvzone.cornerRect(imgBackGround, bbox=bbox, rt=0)
                id = studentsID[matchIdx]
                if counter == 0:
                    counter = 1
                    modeType = 1 # Show face and information mode

        if counter != 0:
            if counter == 1:
                studentsInfo = db.reference(f'Students/{id}').get()
                print(studentsInfo)
                # Get the image from the storage
                blob = bucket.get_blob(f'Images/{id}.png')
                array = np.frombuffer(blob.download_as_string(), np.uint8)
                imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)

                # Update data of attendance

                date_time = datetime.strptime(studentsInfo['last_attendance_time'],
                                             '%d-%m-%Y %H:%M:%S')

                # Check time since last attendance
                timeElapsed = (datetime.now() - date_time).total_seconds()

                # If time elapsed then check attendance again
                if timeElapsed > 30:
                    ref = db.reference(f'Students/{id}')
                    studentsInfo['total_attendance'] += 1
                    ref.child('total_attendance').set(studentsInfo['total_attendance'])
                    ref.child('last_attendance_time').set(datetime.now().strftime('%d-%m-%Y %H:%M:%S'))

                # Else wait
                else:
                    modeType = 3 # Already marked mode
                    counter = 0
                    imgBackGround[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

            if modeType != 3:

                if 10 < counter < 20:
                    modeType = 2 # Active mode

                imgBackGround[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                # Show all the student information
                if counter <= 10:

                    cv2.putText(imgBackGround, str(studentsInfo['total_attendance']), (861, 125),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

                    cv2.putText(imgBackGround, str(studentsInfo['major']), (1006, 550),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

                    cv2.putText(imgBackGround, str(id), (1006, 493),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

                    cv2.putText(imgBackGround, str(studentsInfo['standing']), (910, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (50, 50, 50), 1)

                    cv2.putText(imgBackGround, str(studentsInfo['year']), (1025, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (50, 50, 50), 1)

                    cv2.putText(imgBackGround, str(studentsInfo['starting_year']), (1125, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (50, 50, 50), 1)

                    (w, h), _ = cv2.getTextSize(studentsInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                    offset = (414 - w) // 2
                    cv2.putText(imgBackGround, str(studentsInfo['name']), (808 + offset, 445),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

                    imgBackGround[175:175 + 216, 909:909 + 216] = imgStudent

                counter += 1

                # Wait for new face to be shown
                if counter >= 20:
                    counter = 0
                    modeType = 0
                    studentsInfo = []
                    imgStudent = []
                    imgBackGround[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    else:
        modeType = 0
        counter = 0

    cv2.imshow("Face Attendance", imgBackGround)

    cv2.waitKey(1)