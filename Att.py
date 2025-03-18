import cv2
import numpy as np
import os
import csv
import time
import pickle


from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(r)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

imgbackground = cv2.imread("bg.png")

COL_NAMES = ['NAME', 'TIME']


while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, dsize=(50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d/%m/%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H/%M/%S")
        exit= os.path.isfile("attendance/attendance_"+date+".csv")
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 20, 255), 2)
        cv2.rectangle(frame, (x, y -40), (x+w, y+h), (50, 50, 255), -1)
        cv2.putText(frame,text=(x,y), org=(x+w,y+h), fontFace=(50,50,255),fontScale=1)
        attendence=str(output[0],str(timestamp))
        imgbackground        
        
        
        


    
    
    