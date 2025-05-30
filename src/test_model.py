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