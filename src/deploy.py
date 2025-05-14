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
        rgb, yolo_res = detector.predict_frame(frame, imgsz=640, conf=0.25)
        rgb = add_overlay(rgb, yolo_res)        
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buf.tobytes() + b"\r\n")
        

# ---------- PROCESAR IMAGEN ----------
def process_image(path: Path):
    img = cv2.imread(str(path))
    rgb, res = detector.predict_frame(img, imgsz=1024, conf=0.25)
    rgb = add_overlay(rgb, res)             
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return io.BytesIO(buf.tobytes()) if ok else None

# ---------- OVERLAY ----------
def add_overlay(img_rgb, yolo_res):
    """
    Escribe "clase 0.83" por cada detección (máx 10 líneas) en la parte superior
    izquierda del frame RGB ya anotado.
    """
    r = yolo_res[0]
    if r.boxes is None:                         # nada detectado
        return img_rgb

    names = detector.model.names               # diccionario id→nombre
    cls  = r.boxes.cls.cpu().numpy().astype(int)
    conf = r.boxes.conf.cpu().numpy()
    lines = [f"{names[c]} {conf[i]:.2f}" for i, c in enumerate(cls)][:10]

    h, w = img_rgb.shape[:2]
    line_h = 30                               # altura entre líneas

    for i, txt in enumerate(lines[::-1]):     # empezamos por la última
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        x = (w - tw) // 2                     # centrado horizontal
        y = h - 10 - i * line_h               # 10 px desde abajo, luego hacia arriba
        cv2.putText(img_rgb, txt, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img_rgb


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
