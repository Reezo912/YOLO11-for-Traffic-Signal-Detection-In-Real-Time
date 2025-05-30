from ultralytics import YOLO, settings
from pathlib import Path
import torch, os

# ── Configuración general ───────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
DATA_PATH = (BASE_DIR / "../data/raw/labels/data.yaml").resolve()

settings.update({"tensorboard": True})
torch.backends.cuda.matmul.allow_tf32 = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# ───────────────────────────── FASE A (0‑90) - 1024 px ────────────────────────────
model = YOLO("yolo11s.pt")
#model = YOLO("./runs/detect/train/weights/last.pt")

results_A = model.train(
    #resume = True,
    # Datos & duración
    data=DATA_PATH,
    imgsz=1024,
    epochs=120,                 # 0‑90 (90 épocas)
    batch=0.8,                  # 80 % de VRAM (dinámico)
    workers=12,

    # Optimizador & LR
    optimizer="SGD",
    lr0=0.01,                  
    momentum=0.937,             
    cos_lr=True,               # Scheduler coseno desde lr0→1e‑5
    warmup_epochs=3,           # warm‑up lineal 3 épocas
    weight_decay=0.0005,        
    cls=1.0,                   

    # Augmentaciones
    mosaic=0.2,               
    close_mosaic=10,           
    mixup=0.15,                
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=5, translate=0.1, scale=0.5,

    label_smoothing=0.1,

    # Otros
    patience=20,              # *Early‑stopping* mAP
    amp=True,
    plots=True,
    seed=42,
)

# ──────────────────────── FASE B (90‑100) - 1280 px (Fine‑tune) ─────────────────────
# Carga los últimos pesos y aumenta la resolución para afinar detalles.
model = YOLO("./runs/detect/train2/weights/last.pt")

results_B = model.train(
    data=DATA_PATH,
    imgsz=1280,                # resolución mayor
    epochs=30,                
    batch=0.8,
    workers=16,

    optimizer="SGD",
    lr0=0.01,                 
    momentum=0.937,
    cos_lr=True,
    warmup_epochs=1,           # evita picos de gradiente al cambiar el tamano del batch
    weight_decay=0.0005,
    cls=1.0,

    # Augmentaciones desactivadas para pulido
    mosaic=0,
    mixup=0,
    copy_paste=0,

    hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
    degrees=5, translate=0.1, scale=0.5,

    label_smoothing=0.1,
    patience=20,
    amp=True,
    save_dir="./runs/detect/fine_tunning"
)

# ───────────────────────────── Validacion final ─────────────────────────────────────
best = YOLO("./runs/detect/fine_tunning/weights/best.pt")
metrics = best.val(data=DATA_PATH, plots=True)
print("mAP50‑95 final:", metrics.box.map)
