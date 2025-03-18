import cv2
import numpy as np
import os
import pickle

# Initialize Video Capture and Cascade Classifier
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Create Data Folder if it Doesn't Exist
if not os.path.exists("data"):
    os.makedirs("data")

face_data = []
i = 0
name = input("Enter your name: ")

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, dsize=(50, 50))

        i += 1
        if len(face_data) <= 100 and i % 10 == 0:
            face_data.append(resized_img)
            cv2.putText(frame, str(len(face_data)), org=(50, 50),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1,
                        color=(255, 255, 255), thickness=2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 20, 255), 1)

    cv2.imshow("frame", frame)
    k = cv2.waitKey(1)
    if len(face_data) == 10: # it counts number of faces 
        break

video.release()
cv2.destroyAllWindows()

# Save Face Data Using Pickle
face_data = np.array(face_data).reshape(100, -1)

if "names.pkl" not in os.listdir("data/"):
    names = [name] * 100
    with open("data/names.pkl", "wb") as f:
        pickle.dump(names, f)
else:
    with open("data/names.pkl", "rb") as f:
        names = pickle.load(f)
    names += [name] * 100
    with open("data/names.pkl", "wb") as f:
        pickle.dump(names, f)

if "face_data.pkl" not in os.listdir("data/"):
    with open("data/face_data.pkl", "wb") as f:
        pickle.dump(face_data, f)
else:
    with open("data/face_data.pkl", "rb") as f:
        faces = pickle.load(f)
    faces = np.concatenate((faces, face_data), axis=0)
    with open("data/face_data.pkl", "wb") as f:
        pickle.dump(faces, f)
