import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

# Initialize Firebase using the provided credentials file, database URL, and storage bucket
cred = credentials.Certificate("./FaceRecRealTimeDB/faceattendancerealtime-25805-firebase-adminsdk-loosv-b83b4282c9.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://faceattendancerealtime-25805-default-rtdb.firebaseio.com/',
    'storageBucket' : 'faceattendancerealtime-25805.appspot.com'
})

# Importing faces images
folderPath = "Images"
PathList = os.listdir(folderPath)
imgList = []
studentsID = []
for path in PathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    studentsID.append(os.path.splitext(path)[0])

    # Upload images to Firebase Storage
    file_name = str(folderPath) + str('/') + str(path)
    bucket = storage.bucket()
    blob = bucket.blob(file_name)
    blob.upload_from_filename(file_name)

print(studentsID)

def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

print("Encoding started ...")
encodeListKnow = findEncodings(imgList)
encodeListKnowWithIDs = [encodeListKnow, studentsID]
print("Encoding complete")

# Save the encoded data using pickle
file = open("./FaceRecRealTimeDB/EncodeFile.p", "wb")
pickle.dump(encodeListKnowWithIDs, file)
file.close()
print("File saved")
