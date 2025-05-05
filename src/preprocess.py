import os
import zipfile
import json
import shutil
from tqdm import tqdm
from PIL import Image
import cv2

# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, '..', 'data', 'raw')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'data', 'MTSD')
ANNOTATIONS_DIR = os.path.join(RAW_DIR, 'annotations')

# Tamaño fijo para YOLO
IMAGE_SIZE = (640, 640)

# Crear estructura de carpetas
def create_folders():
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

# Descomprimir todos los zip de imágenes
def unzip_files():
    for file in os.listdir(RAW_DIR):
        if file.endswith('.zip') and 'annotation' not in file:
            zip_path = os.path.join(RAW_DIR, file)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(RAW_DIR)

# Determinar a qué split pertenece una imagen
def get_split_from_filename(filename):
    if 'test' in filename:
        return 'test'
    elif 'val' in filename:
        return 'val'
    else:
        return 'train'

# Procesar una imagen y sus anotaciones
def process_image(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    image_filename = data['image']['filename']
    image_path = os.path.join(RAW_DIR, image_filename)

    if not os.path.exists(image_path):
        print(f"Imagen no encontrada: {image_path}")
        return

    # Cargar y redimensionar imagen
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error leyendo imagen: {image_path}")
        return
    img_resized = cv2.resize(img, IMAGE_SIZE)

    split = get_split_from_filename(image_filename)
    new_image_path = os.path.join(OUTPUT_DIR, 'images', split, image_filename)

    # Guardar imagen redimensionada
    cv2.imwrite(new_image_path, img_resized)

    # Crear archivo .txt de anotaciones
    label_path = os.path.join(OUTPUT_DIR, 'labels', split, os.path.splitext(image_filename)[0] + '.txt')
    img_height, img_width = img.shape[:2]

    with open(label_path, 'w') as label_file:
        for annotation in data['annotations']:
            bbox = annotation['bbox']
            category_id = annotation['category_id']

            x_min, y_min, width, height = bbox
            x_center = (x_min + width / 2) / img_width
            y_center = (y_min + height / 2) / img_height
            width_norm = width / img_width
            height_norm = height / img_height

            label_file.write(f"{category_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")

def main():
    print("🔧 Creando carpetas de salida...")
    create_folders()

    print("🗜️ Descomprimiendo archivos zip de imágenes...")
    unzip_files()

    print("🖼️ Procesando imágenes y anotaciones...")
    annotation_files = [os.path.join(ANNOTATIONS_DIR, f) for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.json')]

    for json_file in tqdm(annotation_files, desc="Procesando"):
        process_image(json_file)

    print("✅ Preprocesamiento completado.")

if __name__ == "__main__":
    main()
