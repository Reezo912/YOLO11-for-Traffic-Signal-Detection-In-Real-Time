from flask import Flask, render_template, Response, request
import cv2
import time

app = Flask(__name__)

def generate_frames(camera_source):
    """Generador de fotogramas desde la fuente dada (IP o webcam local)."""
    cap = cv2.VideoCapture(camera_source)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara con fuente: {camera_source}")

    while True:
        success, frame = cap.read()
        if not success:
            continue

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        time.sleep(0.03)

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    ip = request.args.get('ip')

    if ip:
        if not ip.startswith("http"):
            # Si solo se ingresó IP:puerto, construye la URL completa
            ip = f"http://{ip}/video"  # o usa /mjpegfeed si es necesario
        camera_source = ip
    else:
        camera_source = 0  # webcam local

    return Response(generate_frames(camera_source), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
