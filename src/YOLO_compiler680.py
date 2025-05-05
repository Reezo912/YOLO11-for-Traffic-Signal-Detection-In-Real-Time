from ultralytics import YOLO, settings
from pathlib import Path

import torch, os

"""
Usare YOLOv11, esta version es compatible con telefonos moviles y plataformas como Raspberri.

Ademas trae una gran cantidad de mejoras respecto a modelos mas antiguos como YOLOv8

En este proyecto no vamos a crear una app movil, pero en caso de retomarlo en un futuro y querer hacerlo, seria posible.

La exportacion del modelo ya entrenado deberia hacerse en coreml para este caso.

Usare la version 's' del modelo, este da mejor rendimiento que el nano, 

Small = 11M parametros
Nano = 3M parametros 

"""

BASE_DIR = Path(__file__).resolve().parent 

DATA_PATH = BASE_DIR / '../data/raw/labels'
DATA_PATH = DATA_PATH.resolve()

LOG_DIR = BASE_DIR

print(DATA_PATH)

model = YOLO("yolo11m.pt")


settings.update({"tensorboard": True})
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
torch.backends.cuda.matmul.allow_tf32 = True

# Fase 1: 680px entrenamiento general
model.train(
    data        = DATA_PATH/"data.yaml",
    imgsz       = 640,
    batch       = -1,          # 16 GB VRAM
    epochs      = 80,
    optimizer   = "SGD",
    lr0         = 0.01,
    lrf         = 0.005,
    cos_lr      = True,
    mosaic      = 0.50,
    copy_paste  = 0.30,
    mixup       = 0.20,
    degrees=5, translate=0.1, scale=0.5, fliplr=0.5,
    weight_decay=5e-4, momentum=0.937,
    patience    = 30,
    seed        = 42,
    plots       = True,
)

# Fase 2: 1024px entrenamiento de optimizacion
model = YOLO("./runs/detect/train/weights/last.pt")
model.train(
    resume      = True,
    imgsz       = 1024,
    batch       = -1,          # auto‑batch: 8 en la 4080
    epochs      = 120,         # +40
    lr0         = 0.006,       # 0.01 × (8/16) × 0.6
    lrf         = 0.0006,
    cos_lr      = True,
    mosaic      = 0,
    copy_paste  = 0,
    mixup       = 0,
    patience    = 30,
)

# Fase 3: 1280px entrenamiento especializado
model = YOLO("./runs/detect/train/weights/last.pt")
model.train(
    resume      = True,
    imgsz       = 1280,
    batch       = -1,           # cabe en 15‑16 GB
    epochs      = 150,         # +30
    lr0         = 0.003,
    lrf         = 0.0003,
    cos_lr      = True,
    mosaic      = 0,
    copy_paste  = 0,
    mixup       = 0,
    label_smoothing = 0.05,
    patience    = 20,
)

# ───── Validación final y exportación ─────────────────────────────
model = YOLO("runs/detect/train/weights/best.pt")
model.val(data=DATA_PATH, plots=True)

# Ejemplo de exportación
# yolo export model=runs/detect/train/weights/best.pt format=coreml int8=True
