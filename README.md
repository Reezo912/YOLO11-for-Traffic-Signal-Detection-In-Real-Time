# Signal Detection in Real-Time

Sube un vídeo o imagen y observa la detección en vivo.

¡Bienvenido a Signal Detection in Real-Time! Este proyecto permite la detección de señales de tráfico en imágenes y vídeo en tiempo real usando YOLO11s optimizado. .

## 🗂️ Estructura del Proyecto

El proyecto está organizado de la siguiente manera:

- **`data/`** → Almacena datasets usados y generados.
- **`src/deploy.py`** → Script principal para lanzar la aplicación (backend).
- **`src/AppTest.py`** → Sirve para testear en local con un proyecto de App que teniamos en mente, sin necesidad de desplegar la web.
- **`src/Model_predictor.py, YOLO_compiler1024.py, etc`** → Módulos y utilidades del pipeline.
- **`src/preprocess.py`** → Procesamiento previo de imágenes.
- **`src/test_model.py`** → Scripts de prueba unitaria y validación de modelos.
- **`src/templates/index.html`** → Página web principal donde puedes subir imágenes o vídeos y ver la inferencia en vivo.
- **`requirements.txt`** → Paquetes Python requeridos por el proyecto.
- **`models/`** → Pesos y checkpoints de los modelos entrenados.

## 🧠 Dataset

Hemos utilizado el dataset MTSD (Mapillary Traffic Sign Dataset), que contiene:

    52,453 imágenes

    257,543 etiquetas de señales de tráfico de diferentes regiones del mundo

Además, realizamos técnicas de augmentation para mejorar la precisión del modelo.

## 🏗️ Arquitectura del Modelo

    YOLO11s → 9.4M parámetros
    Entrenado desde cero por 120 epochs
    Fine tuning a resolución 1280×1280 por 30 epochs adicionales

## 📊 Resultados Obtenidos

    mAP50: 0.62
    mAP95: 0.50
    Precisión: 0.74
    Recall: 0.55

Resultados iniciales:

    mAP50: 0.3
    Recall: 0.23
    Precisión: 0.45

## ⚙️ Pipeline en Producción

    OpenCV captura cada frame o recibe la imagen/vídeo a analizar.
    El frame se redimensiona a 1024×1024.
    YOLO11s realiza inferencia: bounding boxes y clases.
    Las señales detectadas se sobreponen en el stream (color por tipo de señal).
    Estadísticas (counts, FPS, etc.) se muestran como un overlay gráfico.

## 🚀 Ejecución de la Aplicación
En linea
   - https://github.com/Reezo912/Detector-Senales-Trafico

En local

  - Clona este repositorio
  - Instala dependencias:
      pip install -r requirements.txt
  - Desde la carpeta raíz o desde src/, ejecuta:
      python src/deploy.py
  - Abre tu navegador en http://127.0.0.1:5000/


Este repositorio ha sido creado como parte del Bootcamp de 4Geeks Academy.
Equipo:

- https://github.com/Reezo912
- https://github.com/JaironMark
- https://github.com/Vic1CR
    
    
Y un especial agradecimiento a nuestros profesores Alessandro Batini y Carlos Vazquez por el tiempo y dedicación en ayudarnos con nuestro proyecto.
