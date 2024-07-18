from flask import Flask, render_template, Response
import cv2
import numpy as np
# นำเข้าฟังก์ชันอื่นๆ ที่จำเป็นจากโค้ดเดิมของคุณ

app = Flask(__name__)
facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

faces_data=[]

i=0
# ฟังก์ชันสำหรับสตรีมวิดีโอ
def gen_frames():
    video = cv2.VideoCapture(0)
    while True:
        success, frame = video.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)