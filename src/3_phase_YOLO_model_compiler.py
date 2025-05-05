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

settings.update({"tensorboard": True})
model = YOLO("yolo11s.pt")

torch.backends.cuda.matmul.allow_tf32 = True
model.model.to(memory_format=torch.channels_last)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"


""" results = model.train(
    data=DATA_PATH/"data.yaml",
    epochs=200,
    imgsz=1024,          
    batch=-1,
    lr0=0.01,           
    lrf=0.01,
    cls=1,              
    dfl=1.5,
    rect=False,
    degrees=5,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=0.2,       
    close_mosaic=50,
    copy_paste=0.15,
    seed=42,
    optimizer='SGD',    
    momentum=0.937,     
    weight_decay=5e-4,  
    patience=20,        
    plots=True,         
) """


""" results = model.train(
    # --- dataset & runtime ---
    data      = DATA_PATH / "data.yaml",
    epochs    = 50,          # mAP suele saturar antes, recorta horas de cómputo
    patience  = 20,
    imgsz     = 1024,
    batch     = 8,           # cabe en 16 GB FP16 (≈ 13 GB reales)
    cache     = "disk",        # acelera lectura de disco
    workers   = 16,           # usa tus núcleos CPU

    # --- optimizador ---
    optimizer = "SGD",
    lr0       = 0.008,        # regla lineal: 0.01 × (batch/16)
    lrf       = 0.01,
    momentum  = 0.937,
    weight_decay = 5e-4,
    warmup_epochs = 3,

    # --- augmentations ---
    mosaic        = 0.20,
    close_mosaic  = 40,       # (int) últimos 40 epochs sin mosaic
    copy_paste    = 0.15,
    cls           = 1.0,
    dfl           = 1.5,
    rect          = False,
    degrees       = 5,
    translate     = 0.10,
    scale         = 0.50,
    fliplr        = 0.5,

    # --- logging / reproducibilidad ---
    seed   = 42,
    plots  = True,
)
 """




from ultralytics import YOLO, settings
from pathlib import Path
import torch, os

# ── Configuración general ────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
DATA_PATH = (BASE_DIR / "../data/raw/labels/data.yaml").resolve()

settings.update({"tensorboard": True})
torch.backends.cuda.matmul.allow_tf32 = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# ──────────── FASE A · 1024 px (0‑50) ───────────────────────────
model = YOLO("yolo11s.pt")
results_A = model.train(
    data      = DATA_PATH,
    imgsz     = 1024,
    epochs    = 50,         
    batch     = 0.8,        # usa el 80 % de la VRAM
    lr0       = 0.008,
    lrf       = 0.001,       # 0.01×lr0
    optimizer = "SGD",
    momentum  = 0.937,
    weight_decay = 5e-4,
    mosaic    = 0.20,
    close_mosaic = 40,       # últimos 10 epochs sin mosaic
    copy_paste = 0.15,
    degrees = 5, translate = 0.10, scale = 0.50, fliplr = 0.5,
    cls = 1.0, dfl = 1.5,
    patience = 20,
    seed = 42,
    plots = True,
    amp = True,
)

# ──────────── FASE B · pulido (50‑80) ───────────────────────────
model = YOLO("runs/detect/train/weights/last.pt")
results_B = model.train(
    resume     = True,
    imgsz      = 1024,
    epochs     = 80,         # +30
    batch      = 0.8,
    lr0        = 0.0008,     # LR ×0.1
    lrf        = 0.00008,
    mosaic     = 0,
    copy_paste = 0,
    close_mosaic = 0,
    patience   = 20,
    amp        = True,
)

# ──────────── FASE C · 1280 px (80‑100) ─────────────────────────
model = YOLO("runs/detect/train/weights/last.pt")  # carga pesos
results_C = model.train(                           # NO resume
    data      = DATA_PATH,
    imgsz     = 1280,
    epochs    = 100,        # +20
    batch     = 0.8,        
    lr0       = 0.0006,     # Escalado para batch efectivo 12
    lrf       = 0.00006,
    mosaic    = 0,
    copy_paste = 0,
    label_smoothing = 0.05,
    patience  = 10,
    amp       = True,
)

# ──────────── Validación final ──────────────────────────────────
best = YOLO("runs/detect/train/weights/best.pt")
metrics = best.val(data=DATA_PATH, plots=True)
print("mAP50‑95 final:", metrics.box.map)



#model.val(imgsz=640)
#model.val(imgsz=1024)