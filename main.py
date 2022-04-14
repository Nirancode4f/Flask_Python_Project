from flask import *
from markupsafe import escape
import cv2
import os

haar_cascade = cv2.CascadeClassifier('PictureData/face_detect.xml')

people = []

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('PictureData/face_trained.yml')


# cv2.imshow("Ben", img)
for i in os.listdir(r'PictureData/face_data'):
    people.append(i)

app = Flask(__name__)



@app.route('/')
def hello_world():  # put application's code here
    return render_template("index.html")


def gen():

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()



        if not ret:
            print("Error: failed to capture image")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ================================ Face_trained reading here ================================

        # Detect the face in the image
        faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces_rect:
            faces_roi = gray[y:y + h, x:x + h]

            label, confidence = face_recognizer.predict(faces_roi)
            print(f'Label = {people[label]} with a confidence of {confidence}')

            cv2.putText(frame, str(people[label]), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

        cv2.imwrite('test_pic.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('test_pic.jpg', 'rb').read() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
        mimetype='multipart/x-mixed-replace; boundary=frame')


@app.errorhandler(404)
def not_found(error):
    return "nothing 404", 404


if __name__ == '__main__':
    app.run()
