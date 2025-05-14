import os, time, cv2
from threading import Thread
from queue import Queue
from flask import Flask, render_template, Response, request, jsonify, send_file
from Model_predictor import SignalDetector
from pathlib import Path
import numpy as np
import io

detector = SignalDetector("models/SignalDetector.pt")

app = Flask(__name__, static_folder="../static", static_url_path="/static")
UPLOAD_DIR = Path(app.root_path) / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# ---------- STREAM DE VÍDEO ----------
q_in = Queue(maxsize=30)          # ~1 s buffer

def generate_frames(src: str | os.PathLike):
    cap = cv2.VideoCapture(str(src), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir: {src}")

    def reader():
        while True:
            ok, f = cap.read()
            if not ok:                                 # loop si acaba
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
            q_in.put(f)
    Thread(target=reader, daemon=True).start()

    while True:
        frame = q_in.get()
        rgb, _ = detector.predict_frame(frame, imgsz=1024, conf=0.25)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buf.tobytes() + b"\r\n")

# ---------- PROCESAR IMAGEN ----------
def process_image(path: Path):
    img = cv2.imread(str(path))
    rgb, _ = detector.predict_frame(img, imgsz=1024, conf=0.25)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return io.BytesIO(buf.tobytes()) if ok else None

# ---------- RUTAS ----------
@app.route("/")
def index(): return render_template("index.html")

@app.post("/upload")
def upload():
    f = request.files.get("file")
    if not f or f.filename == "":  # sirve para jpg/png/mp4
        return jsonify({"error": "no file"}), 400
    name = f"{int(time.time())}_{f.filename}"
    dest = UPLOAD_DIR / name
    f.save(dest)
    return jsonify({"file": name})

@app.route("/video")
def video():
    file = request.args.get("file", "")
    path = UPLOAD_DIR / file
    if not path.exists(): return "missing", 404
    return Response(generate_frames(path),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/image")
def image():
    file = request.args.get("file", "")
    path = UPLOAD_DIR / file
    if not path.exists(): return "missing", 404
    buf = process_image(path)
    return send_file(buf, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(debug=True)
