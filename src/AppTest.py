#!/usr/bin/env python3
"""
Interfaz mínima PySide6 para probar tu modelo YOLO en un vídeo local.

Requisitos:
    pip install PySide6 opencv-python ultralytics

Uso:
    python traffic_sign_gui.py  # se abrirá la ventana
    ▸ Pulsa "Cargar vídeo" y elige un .mp4
    ▸ El vídeo se reproduce anotado en tiempo (casi) real

Ajustes:
    - Cambia MODEL_PATH para tu ubicación.
    - Toca IMG_SIZE, CONF o IOU si quieres otros valores.
"""
from pathlib import Path
import time

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# ────────────────────────────────────────────────────────────────
#  AJUSTES RÁPIDOS
# ────────────────────────────────────────────────────────────────
MODEL_PATH = Path("./models/last.pt")   # cambia si lo tienes en otro sitio
IMG_SIZE   = 640                        # resolución de entrada YOLO
CONF       = 0.25                        # umbral de confianza
IOU        = 0.7                       # NMS IoU
DEVICE     = "cpu"                      # "mps" en Mac M‑series; "cpu" si fallase

# ────────────────────────────────────────────────────────────────
#  MODELO
# ────────────────────────────────────────────────────────────────
try:
    from Model_predictor import SignalDetector  # tu clase
except ImportError:
    raise SystemExit("No se encontró prediction_app.SignalDetector. Asegúrate de que está en PYTHONPATH.")

print("Cargando modelo… (puede tardar unos segundos)")
DETECTOR = SignalDetector(MODEL_PATH, [3])
print("Modelo cargado ✔")

# ────────────────────────────────────────────────────────────────
#  HILO DE VÍDEO
# ────────────────────────────────────────────────────────────────
class VideoThread(QThread):
    frame_ready = Signal(np.ndarray)  # frame RGB anotado

    def __init__(self, video_path: str, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self._running = True

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        delay = 1.0 / fps
        while self._running:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            annotated, _ = DETECTOR.predict_frame(frame, imgsz=IMG_SIZE, conf=CONF, iou=IOU)
            self.frame_ready.emit(annotated)
            # Espera el tiempo restante hasta el siguiente frame
            elapsed = time.time() - t0
            to_wait = delay - elapsed
            if to_wait > 0:
                time.sleep(to_wait)
        cap.release()

    def stop(self):
        self._running = False
        self.wait()

# ────────────────────────────────────────────────────────────────
#  UI PRINCIPAL
# ────────────────────────────────────────────────────────────────
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Traffic‑Sign Demo")
        self.setGeometry(100, 100, 900, 600)

        self.label = QLabel("Carga un vídeo para empezar")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 18px; border: 1px dashed gray;")

        self.btn_load = QPushButton("Cargar vídeo")
        self.btn_load.clicked.connect(self.load_video)

        layout = QVBoxLayout(self)
        layout.addWidget(self.label, stretch=1)
        layout.addWidget(self.btn_load)

        self.thread: VideoThread | None = None

    # ──────────────────────────────────────────────
    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Selecciona un vídeo", "", "Videos (*.mp4 *.avi *.mov)")
        if not file_path:
            return
        # Detener un hilo previo si existía
        if self.thread and self.thread.isRunning():
            self.thread.stop()
        self.thread = VideoThread(file_path)
        self.thread.frame_ready.connect(self.update_frame)
        self.thread.start()

    # ──────────────────────────────────────────────
    def update_frame(self, frame_rgb: np.ndarray):
        h, w, _ = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimg))  

    # ──────────────────────────────────────────────
    def closeEvent(self, event):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
        event.accept()

# ────────────────────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
