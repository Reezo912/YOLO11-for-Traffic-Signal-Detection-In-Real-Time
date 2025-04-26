import os
import json
import tqdm    # libreria para mostrar progresos en python

from pathlib import Path    # con esto referenciamos a la ruta del proyecto


"""
Este script nos permite convertir las anotaciones del dataset de formato COCO -> Common Objects in Context
a formato YOLO.

El dataset ya tiene divididos las imagenes en train/validation/test y nos facilita unos .txt junto con las annotations,
usaremos estos .txt para distribuir las anotaciones entre cada una de nuestras clases. 

"""

# Configurar rutas

# Crear carpetas /labels/train, /val, /test

# Para cada split (train, val, test):
#     Cargar lista de imágenes del split

#     Para cada imagen del split:
#         Cargar su json de anotaciones
#         Leer width y height de la imagen

#         Para cada objeto anotado:
#             Leer label
#             Leer bbox (xmin, ymin, xmax, ymax)
#             Convertir bbox a (x_center, y_center, w, h) normalizados
#             Mapear label a id numérico
#             Escribir línea en formato YOLO

# Guardar todas las líneas en un archivo .txt por imagen

# Crear data.yaml al final


BASE_DIR = Path(__file__).resolve().parent   # carpeta donde esta el script


DATA_DIR = BASE_DIR / '../data/raw'  # indico la ruta de mis datos raw
DATA_DIR = DATA_DIR.resolve()                # lo convierto en una ruta absoluta limpia

lista_tipos_datos = ['labels', 'images']
lista_split = ['train', 'val', 'test']


# creo las 3 carpetas en las que dividire mis labels
for dato in lista_tipos_datos:
    for carpeta in lista_split:
        (DATA_DIR / dato / carpeta).mkdir(parents=True, exist_ok=True)
"""Se que es un doble bucle pero no hay riesgo de que sea infinito. Es horrible pero efectivo"""

SPLITS_DIR = DATA_DIR / './COCO_Annotation/splits'
SPLITS_DIR = SPLITS_DIR.resolve()

for name in lista_split:
    split_file = SPLITS_DIR / (name+'.txt')
    with open(split_file, 'r') as f:
        img_id = [line.strip() for line in f]

print(img_id)