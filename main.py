from flask import *
from markupsafe import escape
import cv2


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

        haar_cascade = cv2.CascadeClassifier('face_detect.xml')

        faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

        # print(f'Number of face found = {len(faces_rect)}')
        for (x, y, w, h) in faces_rect:
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
    return render_template('404.html'), 404


if __name__ == '__main__':
    app.run()
