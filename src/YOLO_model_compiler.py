from ultralytics import YOLO
from pathlib import Path

"""
Usare YOLOv8, esta version, si bien no es la mas moderna, es compatible con telefonos moviles como el iPhone 14 pro.

En este proyecto no vamos a crear una app movil, pero en caso de retomarlo en un futuro y querer hacerlo, seria posible.

La exportacion del modelo ya entrenado deberia hacerse en coreml para este caso.

Usare la version 's' del modelo, este da mejor rendimiento que el nano, 

Small = 11M parametros
Nano = 3M parametros 

"""

BASE_DIR = Path(__file__).resolve().parent 

data_path = BASE_DIR / '../data/raw/labels'
data_path = data_path.resolve()


print(data_path)

model = YOLO("yolov8s.pt")

# Train the model
results = model.train(
    data=data_path/"data.yaml", 
    epochs=100,
    imgsz=640,
    batch=32,
    lr0=0.001,
    lrf=0.01,
    cls=0.7,  # valor aumentado por desbalanceo de clases, default 0.5
    dfl=1.5,
    degrees=5,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    copy_paste=0,
    seed=42,
    optimizer='AdamW',
    cache='disk',
    project="runs",
    name="senales‑v1",
    loggers="tensorboard",
    )