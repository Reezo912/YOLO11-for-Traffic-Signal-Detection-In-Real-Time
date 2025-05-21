import os, time
from threading import Thread
from queue import Queue

import cv2
from flask import Flask, render_template, Response, request, jsonify

from Model_predictor import SignalDetector
detector = SignalDetector("./models/SignalDetector.pt")

app = Flask(__name__, static_folder="../static", static_url_path="/static")
UPLOAD_DIR = os.path.join(app.root_path, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

q_in = Queue(maxsize=30)  # ~1 s de búfer a 30 FPS

# ────────────────────────────────────────────────
def generate_frames(src):
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente: {src}")

    # lector en hilo aparte (solo decodifica)
    def reader():
        while True:
            ok, f = cap.read()
            if not ok:
                if cap.get(cv2.CAP_PROP_POS_FRAMES) >= cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                ):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            q_in.put(f)

    Thread(target=reader, daemon=True).start()

    # bucle principal: inferencia GPU + JPEG OpenCV
    while True:
        frame = q_in.get()  # bloquea hasta tener frame
        annotated_rgb, _ = detector.predict_frame(frame, imgsz=1024, conf=0.25)
        bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            + buf.tobytes()
            + b"\r\n"
        )


# ────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.post("/upload")
def upload():
    f = request.files.get("video")
    if not f or f.filename == "":
        return jsonify({"error": "no file"}), 400
    name = f"{int(time.time())}_{f.filename}"
    f.save(os.path.join(UPLOAD_DIR, name))
    return jsonify({"file": name})


@app.route("/video")
def video():
    file = request.args.get("file", "").strip()
    ip = request.args.get("ip", "").strip()
    cam = request.args.get("camera_id", "").strip()

    if file:
        src = os.path.join(UPLOAD_DIR, file)
    elif ip:
        src = f"http://{ip}/video" if not ip.startswith("http") else ip
    elif cam.isdigit():
        src = int(cam)
    else:
        src = 0

    return Response(
        generate_frames(src),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    app.run(debug=True)
