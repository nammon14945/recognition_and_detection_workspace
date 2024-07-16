from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

with open('model/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('model/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

COL_NAMES = ['NAME', 'TIME']

run_time = 60  # Run for 60 seconds
start_time = time.time()

while time.time() - start_time < run_time:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50,50)).flatten().reshape(1,-1)
        output = knn.predict(resized_img)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")
        print(f"Detected: {output[0]} at {timestamp}")
        attendance = [str(output[0]), str(timestamp)]
        
        # Write attendance every 5 seconds
        if int(ts) % 5 == 0:
            print("Attendance Taken..")
            if exist:
                with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(attendance)
            else:
                with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(COL_NAMES)
                    writer.writerow(attendance)

video.release()
print("Program finished")