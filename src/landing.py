import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile
from pathlib  import Path
from prediction_app import SignalDetector
import numpy as np
import cv2 as cv
import os
import time


# Configuración de la página
st.set_page_config(page_title="Decode Traffic Signs", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / '../models'
MODEL_DIR = MODEL_DIR.resolve()

model_version = "best.pt"

# Cargar modelo
model = YOLO(MODEL_DIR / model_version)

# Estilo CSS para botón Streamlit real
st.markdown("""
    <style>
    .stApp { background-color: #000000; padding: 2rem; }
    div[data-testid="stVerticalBlock"] {
        background-color: #FFFAF0;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 0 10px rgba(0,0,0,0.5);
    }
    .info-box, .file-uploader-box {
        flex-grow: 1;
        padding: 20px;
        border-radius: 10px;
        box-sizing: border-box;
    }
    .info-box { background-color: #f0f2f6; }
    .file-uploader-box {
        background-color: #ffffff;
        border: 2px dashed #ccc;
        text-align: center;
    }
    .file-uploader-box > div {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
    }

    /* Botón personalizado */
    div.stButton > button {
        background-color: #1f77b4;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 8px;
        font-size: 16px;
        cursor: pointer;
        margin: 20px auto;
        display: block;
    }
    </style>
""", unsafe_allow_html=True)

# Contenido principal
with st.container():
    st.markdown("<h1>🚦 Decode Traffic Signs</h1>", unsafe_allow_html=True)
    st.markdown("""
        <div class='intro-text'>
        Este modelo <strong>YOLOv8</strong> te ayudará a identificar y clasificar señales de tráfico,
        las cuales categorizará en las siguientes clases detectadas:
        <strong>Advertencia</strong>, <strong>Prohibición</strong>, <strong>Obligación</strong> e <strong>Información</strong>.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📸 Subir una imagen")

    col1, col2 = st.columns(2)

    with col1:
            # Ruta de guardado
            save_dir = "./data/processed"
            os.makedirs(save_dir, exist_ok=True)

            # Borrado de ficheros en ruta de guardado (como si fuese una cache)
            for filename in os.listdir(save_dir):
                file_path = os.path.join(save_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            # Carga del archivo y guardado en disco
            uploaded_file = st.file_uploader("Admite fotos o videos", type=["jpg", "jpeg", "png", "mp4"])
            if uploaded_file is not None:
                save_path = os.path.join(save_dir, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Debug save {save_path}")

                if uploaded_file is not None:
                    if uploaded_file.type.startswith("image"):
                        image = Image.open(uploaded_file)
                        st.image(image, caption="Imagen original", use_container_width=True)
        

    with col2:
        st.markdown("""
    <style>
    .description-box {
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 4px;
        background: #f9f9f9;
        margin-bottom: 1rem;
    }
    .description-box p {
        margin: 0;
        font-size: 0.95rem;
        color: #333;
    }
    </style>

    <div class='description-box'>
        <p>El modelo hará un análisis del archivo y aquí podrás verlo en tiempo real.</p>
    </div>
""", unsafe_allow_html=True)

            # Boton para analizar el archivo
        if st.button("🔍 Analizar") and uploaded_file is not None:
            detector = SignalDetector(MODEL_DIR / model_version)

            # ───── Procesar IMAGEN ───────────────────────────
            if uploaded_file.type.startswith("image"):
                img = Image.open(save_path)              # PIL
                rgb = np.asarray(img.convert("RGB"))     # RGB
                bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR) # YOLO quiere BGR
                out = detector.predict_frame(bgr, imgsz=1024, conf=0.5)
                st.image(out, caption="Resultado", use_container_width=True)




# TODO sustituir el video en tiempo real por procesado de video y poder descargarlo posteriormente.



            # ───── Procesar VÍDEO “en tiempo real” ───────────
            elif uploaded_file.type.startswith("video"):
                cap = cv.VideoCapture(save_path)
                stframe   = st.empty()                # contenedor para la imagen
                progress  = st.progress(0)            # barra opcional
                total     = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

                frame_id = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    annotated = detector.predict_frame(frame, imgsz=1024, conf=0.6)
                    stframe.image(annotated, channels="RGB", use_container_width=True)

                    frame_id += 1
                    progress.progress(frame_id / total)

                    # Ajusta a tu gusto: 0 = lo más rápido que aguante la CPU/GPU
                    # o sleep(1/fps) para respetar la velocidad del vídeo original
                    cv.waitKey(1)

                cap.release()
                progress.empty()
            
            

    with st.expander("⬇️ Más info acerca de este proyecto"):
        st.markdown("""
        Este proyecto utiliza el modelo <strong>YOLOv11</strong> entrenado con un dataset de señales de tráfico.
        Está diseñado para funcionar en aplicaciones interactivas como esta, permitiendo detección en imágenes de forma rápida.
        Se implementó usando <strong>Streamlit</strong> para la interfaz y <strong>OpenCV</strong> para el procesamiento de imágenes.
        """, unsafe_allow_html=True)

# El bloque 'with st.container():' cierra automáticamente el contenedor