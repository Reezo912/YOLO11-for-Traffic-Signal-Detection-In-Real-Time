from pathlib import Path
import os

import os

os.environ.pop("CUDA_VISIBLE_DEVICES", None)

import torch  # tras limpiar, comprobamos disponibilidad
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


from ultralytics import YOLO


model_subdir = '../models/best.pt'

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = (BASE_DIR / model_subdir) .resolve()



model = YOLO(MODEL_DIR)

# Formato ONNX multiplataforma
model.export(    
    format="onnx",
    opset=12,            # Compatibilidad con ONNX Runtime
    simplify=True,       # Limpia la red para hacerla más ligera
    imgsz=1024,          # Tamaño base (se puede cambiar luego si usas entrada dinámica)
    # nms=True,          # True para evitar la post-pro manual
    device = 'cpu'
    )

# Formato CoreML para MacOs
model.export(
    format="coreml",
    opset=12,            # Compatibilidad con ONNX Runtime
    simplify=True,       # Limpia la red para hacerla más ligera
    imgsz=1024,           # Tamaño base (se puede cambiar luego si usas entrada dinámica)
    # nms=True,           # True para evitar la post-pro manual
    device = 'cpu'
)

# Formato TensorRT para usar graficas Nvidia
model.export(
    format="engine",
    imgsz=1024,
    half=True,                  # Usa FP16 si GPU lo soporta
    device = 'cuda:0',
)

from ultralytics.utils.benchmarks import benchmark
benchmark(model=YOLO(MODEL_DIR / "best.onnx"))