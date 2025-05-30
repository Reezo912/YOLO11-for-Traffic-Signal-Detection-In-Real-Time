import os, time, cv2
from threading import Thread, Event # Importar Event
from queue import Queue, Full, Empty # Importar Full para excepciones de cola
from flask import Flask, render_template, Response, request, jsonify, send_file
from Model_predictor import SignalDetector
from pathlib import Path
import numpy as np
import io



detector = None

app = Flask(__name__, static_folder="../static", static_url_path="/static")
UPLOAD_DIR = Path(app.root_path) / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# ---------- STREAM DE VÍDEO ----------
q_in = Queue(maxsize=30)          # ~1 s buffer

def get_detector():
    global detector
    if detector is None:
        detector = SignalDetector("models/SignalDetector.pt")
    return detector

def generate_frames(src: str | os.PathLike):
    cap = cv2.VideoCapture(str(src), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        app.logger.error(f"No se pudo abrir el vídeo: {src}")
        # Podrías generar un fotograma de error aquí si lo deseas
        return

    # Recursos locales para esta petición de vídeo específica
    frame_queue = Queue(maxsize=30)  # ~1s buffer, ahora local
    stop_event = Event()             # Evento para detener el hilo lector

    def reader_thread_func(video_capture, queue, stop_event_flag, video_source_path):
        app.logger.info(f"Iniciando hilo lector para: {video_source_path}")
        try:
            while not stop_event_flag.is_set():
                ok, frame = video_capture.read()
                if not ok:
                    app.logger.info(f"Vídeo finalizado o error de lectura para {video_source_path}. Reiniciando.")
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = video_capture.read() # Intenta leer de nuevo
                    if not ok:
                        app.logger.warning(f"Fallo al leer fotograma después de reiniciar para {video_source_path}. Deteniendo hilo lector.")
                        break # Detener si el reinicio falla
                
                if frame is not None:
                    try:
                        # Poner fotograma con timeout para evitar bloqueo si el consumidor se detiene
                        queue.put(frame, timeout=1) 
                    except Full:
                        if stop_event_flag.is_set(): # Si nos estamos deteniendo, está bien
                            break
                        app.logger.warning(f"Cola llena para {video_source_path}, el consumidor podría estar lento.")
                        pass # Continuar intentando o romper, según el comportamiento deseado
                else: # frame is None (inesperado si ok es True)
                    app.logger.warning(f"Se leyó un fotograma None de {video_source_path} a pesar de ok=True. Tratando como error/fin.")
                    break
        except Exception as e:
            app.logger.error(f"Excepción en el hilo lector para {video_source_path}: {e}")
        finally:
            app.logger.info(f"Liberando captura de vídeo para {video_source_path} en el hilo lector.")
            video_capture.release()
            # Señalizar al consumidor que no se producirán más fotogramas
            try:
                queue.put(None, timeout=0.1) # Poner un valor centinela (None)
            except Full:
                app.logger.warning(f"No se pudo poner el centinela en la cola para {video_source_path} al salir el lector.")
        app.logger.info(f"Hilo lector para {video_source_path} finalizado.")

    # Iniciar el hilo lector dedicado
    # Pasamos src como video_source_path para logging dentro del hilo
    thread = Thread(target=reader_thread_func, args=(cap, frame_queue, stop_event, str(src)), daemon=True)
    thread.start()

    try:
        while True:
            try:
                frame = frame_queue.get(timeout=1) # Obtener fotograma con timeout
            except Empty: # queue.Empty (si importas `from queue import Empty`)
                if not thread.is_alive() and frame_queue.empty():
                    app.logger.info(f"Hilo lector para {src} ha muerto y la cola está vacía. Deteniendo stream.")
                    break
                continue # Continuar esperando fotogramas si la cola está temporalmente vacía pero el hilo vive

            if frame is None: # Valor centinela del hilo lector indica que ha terminado
                app.logger.info(f"Recibido centinela del lector para {src}. Finalizando stream.")
                break

            # Procesar fotograma
            current_detector = get_detector() # Obtener instancia del detector
            rgb, yolo_res = current_detector.predict_frame(frame, imgsz=640, conf=0.25)
            rgb = add_overlay(rgb, yolo_res) # add_overlay debe ser definida o importada        
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
            
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                   buf.tobytes() + b"\r\n")
    except GeneratorExit:
        app.logger.info(f"Cliente desconectado del stream de vídeo: {src}")
    except Exception as e:
        app.logger.error(f"Error en generate_frames para {src}: {e}")
    finally:
        app.logger.info(f"Deteniendo hilo lector para {src}...")
        stop_event.set() # Señalizar al hilo lector que se detenga
        thread.join(timeout=5) # Esperar a que el hilo lector termine (con un timeout)
        if thread.is_alive():
            app.logger.warning(f"El hilo lector para {src} no se detuvo a tiempo.")
        
        # Vaciar la cola para liberar memoria por si quedó algo
        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except Empty: # queue.Empty
                break
        app.logger.info(f"generate_frames para {src} finalizado.")
        

# ---------- PROCESAR IMAGEN ----------
def process_image(path: Path):
    img = cv2.imread(str(path))
    rgb, res = get_detector().predict_frame(img, imgsz=640, conf=0.25)
    rgb = add_overlay(rgb, res)             
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 65])
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
