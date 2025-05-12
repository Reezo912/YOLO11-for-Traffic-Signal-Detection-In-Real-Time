from pathlib import Path
from ultralytics import YOLO



BASE_DIR = Path(__file__).resolve().parent 

YAML_DIR = BASE_DIR / '../data/raw/labels'
YAML_DIR.resolve()

MODEL_DIR = BASE_DIR/ '../models'
MODEL_DIR.resolve()


model = YOLO('./runs/detect/fine_tunning/weights/last.pt')
metrics = model.val(data=YAML_DIR/'data.yaml', split='val', imgsz=1024)

print(metrics)



from ultralytics.utils.benchmarks import benchmark
benchmark(model=YOLO(MODEL_DIR / "best.pt", data = YAML_DIR))
""" benchmark(model=YOLO(MODEL_DIR / "best.onnx", data = YAML_DIR))
benchmark(model=YOLO(MODEL_DIR / "best.mlpackage/Data/com.apple.CoreML/model.mlmodel", data = YAML_DIR))
benchmark(model=YOLO(MODEL_DIR / "best.engine", data = YAML_DIR)) """