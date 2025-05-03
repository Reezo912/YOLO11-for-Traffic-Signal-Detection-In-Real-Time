from pathlib import Path
from ultralytics import YOLO



BASE_DIR = Path(__file__).resolve().parent 

YAML_DIR = BASE_DIR / '../data/raw/labels'
YAML_DIR.resolve()

MODEL_DIR = BASE_DIR/ '../runs/detect/train/weights'
MODEL_DIR.resolve()


model = YOLO(MODEL_DIR/'best.pt')
metrics = model.val(data=YAML_DIR/'data.yaml', split='val')

print(metrics)