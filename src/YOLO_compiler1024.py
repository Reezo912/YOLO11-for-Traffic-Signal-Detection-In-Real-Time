"""Optimized training script for YOLOv11s on MTSD (fully‑annotated images)

Objetivo
========
- Mantener la arquitectura YOLOv11s y las resoluciones 1024 → 1280 px.
- Conservar `batch = 0.8` (uso dinámico del 80 % de la VRAM).
- Mejorar mAP50‑95 y precisión, reduciendo falsos positivos.
- No superar ~20 h totales (≈ 100 epochs × 16 min/epoch).

Cambios clave (versión SGD)
---------------------------
1. **SGD** + *momentum* 0.90 y *warm‑up* lineal 3 épocas.
2. **Cosine LR** desde 0.01 → 1e‑5 (fase A) y 0.002 → 1e‑5 (fase B).
3. **Weight decay 5e‑4** (igual que en AdamW).
4. **cls = 1.0** para equilibrar la pérdida de clasificación.
5. Resto de ajustes idénticos a la versión original.
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
#model = YOLO("yolo11s.pt")
model = YOLO("./runs/detect/train2/weights/last.pt")

results_A = model.train(
    resume = True,
    # Datos & duración
    data=DATA_PATH,
    imgsz=1024,
    epochs=90,                 # 0‑90 (90 épocas)
    batch=-1,                  # 80 % de VRAM (dinámico)
    workers=16,

    # Optimizador & LR
    optimizer="SGD",
    lr0=0.01,                  # LR base para SGD
    momentum=0.937,             # probar también 0.937 en un grid‑search
    cos_lr=True,               # Scheduler coseno desde lr0→1e‑5
    warmup_epochs=0,           # warm‑up lineal 3 épocas
    weight_decay=5e-4,         # igual que antes
    cls=1.0,                   # peso de la pérdida de clasificación

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
    amp=False,
    plots=True,
    seed=42,
)

# ──────────────────────── FASE B (90‑100) · 1280 px (Fine‑tune) ─────────────────────
# Carga los últimos pesos y aumenta la resolución para afinar detalles.
model = YOLO("./runs/detect/train2/weights/last.pt")

results_B = model.train(
    data=DATA_PATH,
    imgsz=1280,                # resolución mayor
    epochs=12,                # 90→100 (+10 épocas)
    batch=6,
    workers=16,

    # Optimizador & LR (SGD continuo)
    optimizer="SGD",
    lr0=0.002,                 # LR base más bajo para fine‑tuning
    momentum=0.937,
    cos_lr=True,
    warmup_epochs=1,           # evita picos de gradiente al cambiar el tamano del batch
    weight_decay=5e-4,
    cls=1.0,

    # Augmentaciones desactivadas para pulido
    mosaic=0,
    mixup=0,
    copy_paste=0,

    label_smoothing=0.1,
    patience=10,
    amp=True,
    save_dir="runs/detect/train3"
)

# ───────────────────────────── Validación final ─────────────────────────────────────
best = YOLO("runs/detect/train/weights/best.pt")
metrics = best.val(data=DATA_PATH, plots=True)
print("mAP50‑95 final:", metrics.box.map)
