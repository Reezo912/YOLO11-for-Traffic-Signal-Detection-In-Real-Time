"""Optimized training script for YOLOv11s on MTSD (fully‑annotated images)

Objetivo
========
- Mantener la arquitectura YOLOv11s y las resoluciones 1024 → 1280 px.
- Conservar `batch = 0.8` (uso dinámico del 80 % de la VRAM).
- Mejorar mAP50‑95 y precisión, reduciendo falsos positivos.
- No superar ~20 h totales (≈ 100 epochs × 16 min/epoch).

Principales mejoras
-------------------
1. **AdamW** + *warm‑up* + *cosine LR* para convergencia estable.
2. **EMA** de pesos durante todo el entrenamiento (decay 0.9998→0.9999).
3. **Mosaic** prolongado + **MixUp** + **Copy‑Paste** (probabilidades moderadas).
4. **Label smoothing = 0.1** constante.
5. **Early‑Stopping** (`patience = 20`).
6. Fine‑tuning a 1280 px solo en las últimas 10 épocas sin augmentaciones pesadas.

Se asume Ultralytics ≥ v8.1, donde estos argumentos están soportados.
"""

from ultralytics import YOLO, settings
from pathlib import Path
import torch, os

# ── Configuración general ───────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
DATA_PATH = (BASE_DIR / "../data/raw/labels/data.yaml").resolve()

settings.update({"tensorboard": True})
torch.backends.cuda.matmul.allow_tf32 = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# ───────────────────────────── FASE A (0‑90) · 1024 px ────────────────────────────
model = YOLO("yolo11s.pt")
 
results_A = model.train(
    # Datos & duración
    data=DATA_PATH,
    imgsz=1024,
    epochs=90,                 # 0‑90 (90 épocas)
    batch=-1,                 # 80 % de VRAM

    # Optimizador & LR
    optimizer="AdamW",
    lr0=0.001,                 # LR base
    cos_lr=True,               # Scheduler coseno desde lr0→1e‑5
    weight_decay=5e-4,         # Decoupled (AdamW)

    # Augmentaciones
    mosaic=0.20,               # prob. 20 % Mosaic
    close_mosaic=10,           # desactiva en las últimas 10 épocas (80‑90)
    mixup=0.15,                # prob. 15 % MixUp
    copy_paste=0.30,           # prob. 30 % Copy‑Paste
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=5, translate=0.10, scale=0.50, fliplr=0.5,

    label_smoothing=0.1,

    # Otros
    patience=20,              # *Early‑stopping* mAP
    amp=True,
    plots=True,
    seed=42,
)

# ──────────────────────── FASE B (90‑100) · 1280 px (Fine‑tune) ─────────────────────
# Carga los últimos pesos y aumenta la resolución para afinar detalles.
model = YOLO("../runs/detect/train2/weights/last.pt")

results_B = model.train(
    resume=True,               # continúa el histórico (epoch 90)
    imgsz=1280,                # resolución mayor
    epochs=100,               # 90→100 (+10 épocas)
    batch=-1,

    # Optimizador & LR más bajo (continúa coseno)
    lr0=0.0005,
    cos_lr=True,
    warmup_epochs=0,           # sin warm‑up adicional

    # Augmentaciones desactivadas para pulido
    mosaic=0,
    mixup=0,
    copy_paste=0,

    label_smoothing=0.1,
    patience=10,
    amp=True,
)

# ───────────────────────────── Validación final ─────────────────────────────────────
best = YOLO("runs/detect/train/weights/best.pt")
metrics = best.val(data=DATA_PATH, plots=True)
print("mAP50‑95 final:", metrics.box.map)
