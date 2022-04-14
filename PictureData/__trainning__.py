import cv2 as cv
import numpy as np
import os

# run this first
# pip install opencv-contrib-python
people = []
features = []
labels = []
dir = "face_data"
haar_cascade = cv.CascadeClassifier("./face_detect.xml")
face_recognizer = cv.face.LBPHFaceRecognizer_create()



# ================================ Save Face Object (LABEL) ================================

def train_running():

    for i in os.listdir(r'face_data'):
        people.append(i)
    #  Read Each Of Picture In Dir
    for person in people:
        path = os.path.join(dir, person)
        label = people.index(person)
        print(path)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            print("img_path ", img_path)
            img = cv.imread(img_path)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

            print(len(faces_rect))

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y + h, x:x + h]
                print("face_roi = ", faces_roi)

                features.append(faces_roi)
                labels.append(label)


# ================================ Save_Face_Reco ================================

train_running()

features = np.array(features, dtype="object")
labels = np.array(labels)

face_recognizer.train(features, labels)
face_recognizer.save("face_trained.yml")