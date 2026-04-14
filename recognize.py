import cv2
import face_recognition
import os
import csv
import pickle
from datetime import datetime

path = 'dataset'
attendance_file = "attendance.csv"
encoding_file = "encodings.pkl"

# 🔹 Load or Create Encodings
def loadEncodings():
    if os.path.exists(encoding_file):
        print("⚡ Loading saved encodings...")
        with open(encoding_file, "rb") as f:
            encodeListKnown, classNames = pickle.load(f)
        return encodeListKnown, classNames

    print("⏳ Encoding faces for first time...")

    images = []
    classNames = []

    for person in os.listdir(path):
        person_path = os.path.join(path, person)

        if not os.path.isdir(person_path):
            continue

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            images.append(img)
            classNames.append(person)

    encodeList = []
    validNames = []

    for img, name in zip(images, classNames):
        img = cv2.resize(img, (0, 0), None, 0.5, 0.5)  # speed boost
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        encodings = face_recognition.face_encodings(img)

        if len(encodings) > 0:
            encodeList.append(encodings[0])
            validNames.append(name)

    # Save encodings
    with open(encoding_file, "wb") as f:
        pickle.dump((encodeList, validNames), f)

    print("✅ Encoding complete & saved")

    return encodeList, validNames


# 🔹 Load encodings
encodeListKnown, classNames = loadEncodings()


# 🔹 Attendance file setup
if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time", "Date"])


# 🔹 Mark Attendance
def markAttendance(name):
    now = datetime.now()
    time = now.strftime('%H:%M:%S')
    date = now.strftime('%Y-%m-%d')

    with open(attendance_file, "r", newline="") as f:
        data = list(csv.reader(f))

    header = data[0]
    records = data[1:]

    for row in records:
        if row[0] == name and row[2] == date:
            return

    records.insert(0, [name, time, date])

    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(records)

    print(f"✅ {name} marked at {time}")


# 🔹 Start Camera
cap = cv2.VideoCapture(0)
window_name = 'Face Recognition Attendance'

# 🔥 Force window to front
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
cv2.resizeWindow(window_name, 800, 600)

print("🎥 Camera Started | Press ESC to exit")

while True:
    success, img = cap.read()
    if not success:
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faces = face_recognition.face_locations(imgS)
    encodes = face_recognition.face_encodings(imgS, faces)

    for encodeFace, faceLoc in zip(encodes, faces):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        distances = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = distances.argmin()

        if matches[matchIndex] and distances[matchIndex] < 0.5:
            name = classNames[matchIndex].upper()
        else:
            name = "UNKNOWN"

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

        color = (0, 255, 0) if name != "UNKNOWN" else (0, 0, 255)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, name, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if name != "UNKNOWN":
            markAttendance(name)

    cv2.imshow(window_name, img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("🛑 Session Ended")