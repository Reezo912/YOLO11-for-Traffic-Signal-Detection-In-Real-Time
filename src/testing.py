import os
import json
from tqdm import tqdm   # libreria para mostrar progresos en python
import typing

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

splits = {}  # inicializo mi diccionario que contendra las ids de train/val/test

for name in lista_split:
    split_file = SPLITS_DIR / (name+'.txt')
    with open(split_file, 'r') as f:
        splits[name] = [ln.strip() for ln in f if ln.strip()]


print(splits.keys())  # compruebo que me este extrayendo correctamente los nombres


"""
En los .json las cajas de Object Location, indican los pixeles absolutos de la imagen para:
    -xmin
    -ymin
    -xmax
    -ymax

YOLO necesita que esten en otro formato tal que: <class_id> <x_center> <y_center> <width> <height>
Con los valores normalizados entre 0 y 1, no pixeles absolutos.

En la siguiente funcion realizare la conversion
    """

def coco2yolo(box: dict, ancho: int, alto: int)-> typing.Tuple[float, float, float, float]:
    """
    Siendo las variables aceptadas box, ancho y alto
        - box: diccionario bbox del archivo .json
        - ancho: dimension x de mi imagen 
        - alto: dimension y de mi imagen
    """
    xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax'] # extraigo las coordenadas del .json
    x_centro = (xmin + xmax) / 2 / ancho      
    y_centro = (ymin + ymax) / 2 / alto      
    wide   = (xmax - xmin) / ancho          
    height   = (ymax - ymin) / alto          
    return x_centro, y_centro, wide, height


"""
En mis .json, tengo el tipo de senal relacionada con un numero.

Tengo que sacar estos tipos junto con su numero y meterlos en un diccionario.

Mas adelante necesitare pasarlos a una lista en la que los indices sean los numeros enteros y esta
estructura de datos es la que le servira a YOLO.

No creo una lista directamente porque seria muy poco efectivo desde un punto de vista computacional,
habria que recorrer dicha lista por cada imagen que me encuentre, buscando si ya he apuntado ese label.
"""


# inicializo mi diccionario que relacionara label con el numero del json
label2id = {}

id2label = []



""" 
Para cada split ('train', 'val', 'test'):
    Para cada imagen en el split:
        Construir la ruta al archivo JSON de la imagen
        Construir la ruta de destino del archivo .txt

        Abrir y leer el JSON de anotaciones
        Obtener el ancho y alto de la imagen

        Inicializar una lista vacía para guardar las líneas YOLO

        Para cada objeto anotado en la imagen:
            Si el objeto es falso (dummy):
                Saltarlo

            Leer el nombre de la clase (label)

            Si la clase no existe en el diccionario:
                Asignarle un nuevo número de clase
                Guardar su nombre en la lista de clases

            Convertir el bounding box a formato YOLO
                (calcular centro x, centro y, ancho, alto normalizados)

            Crear una línea con el formato YOLO
            Añadir esa línea a la lista

        Escribir todas las líneas en un archivo .txt correspondiente a la imagen

 """

ANN_DIR = DATA_DIR / './COCO_Annotation/annotations/'
ANN_DIR =  ANN_DIR.resolve()


for split, ids in splits.items():
    if split == 'test':  # tengo que ignorar el test, no contiene etiquetas y da error
        continue

    for img_id in tqdm(ids, desc=f'{split:5}', ncols=80):  # tqdm es una libreria para mostrar una barra de progreso, ids seria donde estamos contando img_id
        ann_path = ANN_DIR / f'{img_id}.json'

        with open(ann_path) as f:
            ann = json.load(f)

        # leo las dimensiones de la imagen en el .json
        ancho, alto = ann['width'], ann['height']

        # inicializo la lista donde guardare los datos
        YOLO_data = []

        for obj in ann['objects']:    # objects es el nombre de la key en el diccionario .json
            label = obj['label']      # extraigo el label

            if label not in label2id:
                label2id[label] = len(label2id)  # voy a usar la longitud de la lista, asi siempre me creara el siguiente integer.
                id2label.append(label)           # lo guardo en mi lista
        
            # extraigo el id de cada foto
            cls_id = label2id[label]

            x_center, y_center, bbox_ancho, bbox_alto = coco2yolo(obj['bbox'], ancho, alto)   # paso la bbox del json y las dimensiones de la foto a mi funcion coco2yolo

            # Creo mis lineas en formato YOLO
            line = f"{cls_id} {x_center:.6f} {y_center:.6f} {bbox_ancho:.6f} {bbox_alto:.6f}"
            YOLO_data.append(line)

        # guardo toda la info transformada en un .txt para  YOLO
        txt_path = DATA_DIR / 'labels' / split / f'{img_id}.txt'
        with open(txt_path, 'w') as f:
            f.write('\n'.join(YOLO_data))


# tengo que crear un archivo yaml donde guardare los datos de mis etiquetas
yaml_path = DATA_DIR / 'labels' / 'data.yaml'
with open(yaml_path, 'w') as f:
    f.write(f"path: {DATA_DIR}\n")
    f.write("train: images/train\nval: images/val\ntest: images/test\n")
    f.write(f"nc: {len(id2label)}\n")
    f.write("names:\n")
    for i, name in enumerate(id2label):
        f.write(f"  {i}: {name}\n")

print("COCO to YOLO conversion finalizada.  Data YAML en:", yaml_path)