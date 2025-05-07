from pathlib import Path
from ultralytics import YOLO



BASE_DIR = Path(__file__).resolve().parent 

YAML_DIR = BASE_DIR / '../data/raw/labels'
YAML_DIR.resolve()

MODEL_DIR = BASE_DIR/ '../runs/detect/train2/weights'
MODEL_DIR.resolve()


model = YOLO(MODEL_DIR/'best.pt')
metrics = model.val(data=YAML_DIR/'data.yaml', split='val', conf=0.25, iou=0.7, imgsz=1024)

print(metrics)