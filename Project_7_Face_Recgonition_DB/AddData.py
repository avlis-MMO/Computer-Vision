import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Initialize Firebase using the provided credentials file and database URL
cred = credentials.Certificate("./cred.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://faceattendancerealtime-25805-default-rtdb.firebaseio.com/'
})

# Get a reference to the 'Students' node in the Firebase Realtime Database
ref = db.reference('Students')

# Data to be added to the database
data = {
    "321654": {
        "name": "Mark Zuckerberg",
        "major": "Engineer",
        "starting_year": 2018,
        "total_attendance": 6,
        "standing": "6",
        "year": 1,
        "last_attendance_time": "08-08-2023 10:00:00"
    },

    "852741": {
        "name": "Emely Blunt",
        "major": "Actress",
        "starting_year": 2016,
        "total_attendance": 12,
        "standing": "8",
        "year": 3,
        "last_attendance_time": "08-08-2023 10:00:00"
    },

    "963852": {
        "name": "Elon Musk",
        "major": "Physics",
        "starting_year": 2017,
        "total_attendance": 8,
        "standing": "4",
        "year": 2,
        "last_attendance_time": "08-08-2023 10:00:00"
    }
}

# Loop through the data and set each key-value pair in the Firebase database
for key, value in data.items():
    ref.child(key).set(value)
