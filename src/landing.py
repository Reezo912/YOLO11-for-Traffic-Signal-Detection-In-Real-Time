# import os
# import json
# from tqdm import tqdm   # libreria para mostrar progresos en python
# import typing
# from pathlib import Path    # con esto referenciamos a la ruta del proyecto
# import streamlit as st


# """
# Este script nos permite convertir las anotaciones del dataset de formato COCO -> Common Objects in Context
# a formato YOLO.

# El dataset ya tiene divididos las imagenes en train/validation/test y nos facilita unos .txt junto con las annotations,
# usaremos estos .txt para distribuir las anotaciones entre cada una de nuestras clases. 

# """

# # Configurar rutas

# # Crear carpetas /labels/train, /val, /test

# # Para cada split (train, val, test):
# #     Cargar lista de imágenes del split

# #     Para cada imagen del split:
# #         Cargar su json de anotaciones
# #         Leer width y height de la imagen

# #         Para cada objeto anotado:
# #             Leer label
# #             Leer bbox (xmin, ymin, xmax, ymax)
# #             Convertir bbox a (x_center, y_center, w, h) normalizados
# #             Mapear label a id numérico
# #             Escribir línea en formato YOLO

# # Guardar todas las líneas en un archivo .txt por imagen

# # Crear data.yaml al final


# BASE_DIR = Path(__file__).resolve().parent
# DATA_DIR = BASE_DIR / '../data/raw'
# DATA_DIR = DATA_DIR.resolve()

# lista_tipos_datos = ['labels', 'images']
# lista_split = ['train', 'val', 'test']


# # creo las 3 carpetas en las que dividire mis labels
# for dato in lista_tipos_datos:
#     for carpeta in lista_split:
#         (DATA_DIR / dato / carpeta).mkdir(parents=True, exist_ok=True)
# """Se que es un doble bucle pero no hay riesgo de que sea infinito. Es horrible pero efectivo"""

# SPLITS_DIR = DATA_DIR / './COCO_Annotation/splits'
# SPLITS_DIR = SPLITS_DIR.resolve()

# splits = {}  # inicializo mi diccionario que contendra las ids de train/val/test

# for name in lista_split:
#     split_file = SPLITS_DIR / (name + '.txt')
#     with open(split_file, 'r') as f:
#         splits[name] = [ln.strip() for ln in f if ln.strip()]


# print(splits.keys())  # compruebo que me este extrayendo correctamente los nombres


# """
# En los .json las cajas de Object Location, indican los pixeles absolutos de la imagen para:
#     -xmin
#     -ymin
#     -xmax
#     -ymax

# YOLO necesita que esten en otro formato tal que: <class_id> <x_center> <y_center> <width> <height>
# Con los valores normalizados entre 0 y 1, no pixeles absolutos.

# En la siguiente funcion realizare la conversion
#     """

# def coco2yolo(box: dict, ancho: int, alto: int)-> typing.Tuple[float, float, float, float]:
#     """
#     Siendo las variables aceptadas box, ancho y alto
#         - box: diccionario bbox del archivo .json
#         - ancho: dimension x de mi imagen 
#         - alto: dimension y de mi imagen
#     """
#     xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax'] # extraigo las coordenadas del .json
#     x_centro = (xmin + xmax) / 2 / ancho      
#     y_centro = (ymin + ymax) / 2 / alto      
#     wide   = (xmax - xmin) / ancho          
#     height   = (ymax - ymin) / alto          
#     return x_centro, y_centro, wide, height


# """
# En mis .json, tengo el tipo de senal relacionada con un numero.

# Tengo que sacar estos tipos junto con su numero y meterlos en un diccionario.

# Mas adelante necesitare pasarlos a una lista en la que los indices sean los numeros enteros y esta
# estructura de datos es la que le servira a YOLO.

# No creo una lista directamente porque seria muy poco efectivo desde un punto de vista computacional,
# habria que recorrer dicha lista por cada imagen que me encuentre, buscando si ya he apuntado ese label.
# """


# # inicializo mi diccionario que relacionara label con el numero del json
# ANN_DIR = DATA_DIR / './COCO_Annotation/annotations/'
# ANN_DIR = ANN_DIR.resolve()

# label2id = {}
# id2label = []


# """ 
# Para cada split ('train', 'val', 'test'):
#     Para cada imagen en el split:
#         Construir la ruta al archivo JSON de la imagen
#         Construir la ruta de destino del archivo .txt

#         Abrir y leer el JSON de anotaciones
#         Obtener el ancho y alto de la imagen

#         Inicializar una lista vacía para guardar las líneas YOLO

#         Para cada objeto anotado en la imagen:
#             Si el objeto es falso (dummy):
#                 Saltarlo

#             Leer el nombre de la clase (label)

#             Si la clase no existe en el diccionario:
#                 Asignarle un nuevo número de clase
#                 Guardar su nombre en la lista de clases

#             Convertir el bounding box a formato YOLO
#                 (calcular centro x, centro y, ancho, alto normalizados)

#             Crear una línea con el formato YOLO
#             Añadir esa línea a la lista

#         Escribir todas las líneas en un archivo .txt correspondiente a la imagen

#  """

# archivos_faltantes = 0  # <-- MUY IMPORTANTE

# for split, ids in splits.items():
#     if split == 'test':
#         continue

#     for img_id in tqdm(ids, desc=f'{split:5}', ncols=80):
#         ann_path = ANN_DIR / f'{img_id}.json'

#         if not ann_path.exists():
#             print(f"[ADVERTENCIA] No se encontró el archivo: {ann_path.name} en la ruta: {ann_path}")
#             archivos_faltantes += 1
#             continue

#         try:
#             with open(ann_path) as f:
#                 ann = json.load(f)
#         except json.JSONDecodeError:
#             print(f"[ADVERTENCIA] Error al decodificar JSON para el archivo: {ann_path.name} en la ruta: {ann_path}")
#             continue
#         except Exception as e:
#             print(f"[ADVERTENCIA] Error inesperado al intentar leer el archivo: {ann_path.name} en la ruta: {ann_path}: {e}")
#             continue

#         if 'width' not in ann or 'height' not in ann:
#             print(f"[ADVERTENCIA] El archivo {ann_path.name} no contiene las claves 'width' o 'height'.")
#             continue

#         # leo las dimensiones de la imagen en el .json
#         ancho, alto = ann['width'], ann['height']

#         # inicializo la lista donde guardare los datos
#         YOLO_data = []

#         for obj in ann['objects']:    # objects es el nombre de la key en el diccionario .json
#             label = obj['label']      # extraigo el label

#             if label not in label2id:
#                 label2id[label] = len(label2id)  # voy a usar la longitud de la lista, asi siempre me creara el siguiente integer.
#                 id2label.append(label)           # lo guardo en mi lista
        
#             # extraigo el id de cada foto
#             cls_id = label2id[label]

#             x_center, y_center, bbox_ancho, bbox_alto = coco2yolo(obj['bbox'], ancho, alto)   # paso la bbox del json y las dimensiones de la foto a mi funcion coco2yolo

#             # Creo mis lineas en formato YOLO
#             line = f"{cls_id} {x_center:.6f} {y_center:.6f} {bbox_ancho:.6f} {bbox_alto:.6f}"
#             YOLO_data.append(line)

#         # guardo toda la info transformada en un .txt para  YOLO
#         txt_path = DATA_DIR / 'labels' / split / f'{img_id}.txt'
#         with open(txt_path, 'w') as f:
#             f.write('\n'.join(YOLO_data))

# print(f"\nProceso terminado ✅")
# print(f"Archivos de anotaciones faltantes: {archivos_faltantes}")

# # tengo que crear un archivo yaml donde guardare los datos de mis etiquetas
# yaml_path = DATA_DIR / 'labels' / 'data.yaml'
# with open(yaml_path, 'w') as f:
#     f.write(f"path: {DATA_DIR}\n")
#     f.write("train: images/train\nval: images/val\ntest: images/test\n")
#     f.write(f"nc: {len(id2label)}\n")
#     f.write("names:\n")
#     for i, name in enumerate(id2label):
#         f.write(f"  {i}: {name}\n")

# print("COCO to YOLO conversion finalizada.  Data YAML en:", yaml_path)


# TESTING STREAMLIT 

# import streamlit as st
# from PIL import Image
# import numpy as np
# import tempfile
# import cv2
# from ultralytics import YOLO
# # Configuración de la página

# st.set_page_config(page_title="Detector de Señales de Tráfico", layout="wide")
# # Título
# st.title(":vertical_traffic_light: Detectar y Clasificar Señales de Tráfico")
# st.markdown("Sube una imagen y el modelo detectará las señales de tráfico presentes en ella.")
# Cargar modelo YOLO
# @st.cache_resource
# def cargar_modelo():
#     modelo = YOLO("/Users/johncarter/Documents/GitHub/Detector-Senales-Trafico/data/raw/labels/data.yaml")  # Cambia por la ruta real de tu modelo entrenado
#     return modelo
# modelo = cargar_modelo()
# Subida de imagen
# uploaded_file = st.file_uploader(":camera: Sube una imagen", type=["jpg", "jpeg", "png"])
# if uploaded_file is not None:
#     imagen = Image.open(uploaded_file).convert("RGB")
#     st.image(imagen, caption=":camera: Imagen subida", use_column_width=True)
#     if st.button(":mag: Detectar Señales"):
#         with st.spinner("Procesando..."):
#             # Guardar temporalmente la imagen
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
#                 imagen.save(tmp.name)
#                 resultados = modelo(tmp.name)
#             # Obtener la imagen con los bounding boxes
#             img_array = resultados[0].plot()  # Devuelve una imagen con las detecciones dibujadas
#             st.image(img_array, caption=":brain: Resultado del modelo", use_column_width=True)
#             # Mostrar tabla de resultados
#             boxes = resultados[0].boxes
#             if boxes is not None and len(boxes) > 0:
#                 clases = boxes.cls.cpu().numpy().astype(int)
#                 confianzas = boxes.conf.cpu().numpy()
#                 nombres = [modelo.names[i] for i in clases]
#                 st.subheader(":bar_chart: Señales detectadas")
#                 for nombre, conf in zip(nombres, confianzas):
#                     st.write(f"- {nombre} ({conf*100:.1f}%)")
#             else:
#                 st.info("No se detectaron señales en la imagen.")
# # Sección informativa
# # Sección informativa
# with st.expander(":information_source: Sobre este proyecto"):
#     st.markdown("""
#     - **Modelo:** YOLOv5/YOLOv8 entrenado con el dataset MTSD (Mapillary Traffic Sign Dataset).
#     - **Objetivo:** Detectar y clasificar señales de tráfico a partir de imágenes.
#     - **Equipo:** Proyecto de grupo en Machine Learning.
#     - **Frameworks:** Streamlit, PyTorch, Ultralytics YOLO.
#     """)

#2nd TRY

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
# # Configuración inicial de la página
# st.set_page_config(page_title="Detector de Señales de Tráfico", layout="wide")
# # Título y descripción
# st.header(":vertical_traffic_light: Detectar y Clasificar Señales de Tráfico")

# st.markdown("""
# Sube una imagen y nuestro modelo entrenado con YOLO detectará y clasificará las señales de tráfico.
# """)

# col1, col2 = st.columns(2)
# col1.write('Column 1')
# col2.write('Column 2')

# # Carga del archivo de imagen
# uploaded_file = st.file_uploader(":camera: Sube una imagen", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     image_np = np.array(image)
#     st.image(image, caption="Imagen subida", use_column_width=True)
#     # Botón para ejecutar detección
#     if st.button(":mag: Detectar Señales"):
#         with st.spinner("Detectando señales..."):
#             # Guardar la imagen temporalmente
#             with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
#                 image.save(tmp_file.name)
#                 image_path = tmp_file.name
#             # ---------- Lógica de detección (YOLO) -----------
#             # Aquí debes ejecutar tu modelo YOLO y guardar la imagen de salida con cajas
#             # Por ejemplo, si usas YOLOv5 con PyTorch:
#             #
#             # from yolov5 import YOLOv5
#             # yolo = YOLOv5("path/to/best.pt", device="cpu")
#             # results = yolo.predict(image_path)
#             # results.save("output.jpg")
#             #
#             # Para el ejemplo, solo copiamos la imagen original
#             output_path = os.path.join(tempfile.gettempdir(), "output.jpg")
#             cv2.imwrite(output_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
#             # Leer imagen de salida
#             output_image = Image.open(output_path)
#             st.success("¡Detección completada!")
#             st.image(output_image, caption="Resultado de la detección", use_column_width=True)
#             # Resultados simulados (reemplaza con tus resultados reales)
#             st.subheader(":bar_chart: Resultados de la detección")
#             st.dataframe({
#                 "Clase": ["Señal de Stop", "Límite 50"],
#                 "Confianza": [0.98, 0.91],
#                 "Posición": [(120, 60, 180, 120), (200, 100, 250, 150)]
#             })
# # Información del modelo
# with st.expander(":information_source: Sobre el modelo"):
#     st.markdown("""
#     - **Modelo**: YOLOv5 entrenado con el dataset MTSD
#     - **Clases detectadas**: advertencia, prohibición, obligación, información
#     - **Framework**: PyTorch + OpenCV
#     """)


    # 3 TRY

import streamlit as st

# Configuración inicial
st.set_page_config(page_title="Detector de Señales de Tráfico", layout="wide")

# Estilos personalizados
st.markdown("""
    <style>
    .stApp {
        background-color: #C0C0C0;
        padding: 2rem;
        display: flex;
        justify-content: center;
    }

    .main-box {
        background-color: white;
        width: 1304px;
        min-height: 580px;
        max-height: 80vh;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
        overflow-y: auto;
    }

    h1 {
        text-align: center;
        color: #ff4b4b;
        font-size: 3em;
        margin-bottom: 0.2em;
    }
    .intro-text {
        text-align: center;
        font-size: 1.2em;
        color: #333333;
        margin-bottom: 2em;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        height: 100%;
    }
    .file-uploader-box {
        background-color: #ffffff;
        border: 2px dashed #ccc;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        height: 100%;
    }
    </style>
""", unsafe_allow_html=True)


# Contenedor blanco de todo el contenido
st.markdown('<div class="main-box">', unsafe_allow_html=True)

# Título centrado
st.markdown("<h1>🚦 Detectar y Clasificar Señales de Tráfico</h1>", unsafe_allow_html=True)

# Texto explicativo centrado
st.markdown("""
<div class='intro-text'>
    Este modelo <strong>YOLOv8</strong> te ayudará a identificar y clasificar señales de tráfico, 
    las cuales categorizará en las siguientes clases detectadas: 
    <strong>Advertencia</strong>, <strong>Prohibición</strong>, <strong>Obligación</strong> e <strong>Información</strong>.
</div>
""", unsafe_allow_html=True)

# Título encima de las columnas
st.markdown("### 📸 Subir una imagen")

# Columnas simétricas
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class='info-box'>
        <strong>YoloV8</strong><br><br>
        Nuestro modelo entrenado con YOLO detectará y clasificará las señales de tráfico presentes en ella.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='file-uploader-box'>
        Usa el siguiente campo para arrastrar y soltar tu imagen o haz clic para seleccionarla.
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

# Desplegable con más información
with st.expander("⬇️ Más info acerca de este proyecto"):
    st.markdown("""
    Este proyecto utiliza el modelo <strong>YOLOv8</strong> entrenado con un dataset de señales de tráfico.  
    Está diseñado para funcionar en aplicaciones interactivas como esta, permitiendo detección en imágenes de forma rápida.  
    Se implementó usando <strong>Streamlit</strong> para la interfaz y <strong>OpenCV</strong> para el procesamiento de imágenes.  
    """, unsafe_allow_html=True)

# Cierre del contenedor principal
st.markdown('</div>', unsafe_allow_html=True)
