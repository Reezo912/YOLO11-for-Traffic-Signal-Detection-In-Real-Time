from flask import Flask, render_template,Response
import cv2
import time 

app=Flask(__name__)
camera=cv2.VideoCapture(0)#"http://192.168.0.21:4747/mjpegfeed")
if not camera.isOpened():
    print("Can not open camera")

def generate_frames():
    while True:
        
        #read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            ret,buffer=cv2.imencode('.jpg', frame)
            frame=buffer.tobytes()
            time.sleep(0.03)
        yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
     return Response(generate_frames() ,mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)

