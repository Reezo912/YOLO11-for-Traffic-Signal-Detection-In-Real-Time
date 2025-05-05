import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
from pathlib  import Path

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
        st.markdown("""
            <div class='info-box'>
                <strong>YoloV8</strong><br><br>
                Nuestro modelo entrenado con YOLO detectará y clasificará las señales de tráfico presentes en ella.
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class='file-uploader-box'>
                <p>Usa el siguiente campo para arrastrar y soltar tu imagen o haz clic para seleccionarla.</p>
            </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_column_width=True)

        # Botón real de Streamlit con estilo
        if st.button("🔍 Analizar"):
            if image.mode == "RGBA":
                image = image.convert("RGB")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                image.save(tmp.name)
                temp_image_path = tmp.name
    
            results = model.predict(source=temp_image_path, save=False, imgsz=1024, conf=0.5)
            result_bgr = results[0].plot()
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            st.image(result_rgb, caption="Resultado del análisis", use_column_width=True)

            st.markdown("### 📋 Detalles de los objetos detectados")
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                st.write(f"🔹 Clase: {model.names[cls]}, Confianza: {conf:.2f}")

    with st.expander("⬇️ Más info acerca de este proyecto"):
        st.markdown("""
        Este proyecto utiliza el modelo <strong>YOLOv8</strong> entrenado con un dataset de señales de tráfico.
        Está diseñado para funcionar en aplicaciones interactivas como esta, permitiendo detección en imágenes de forma rápida.
        Se implementó usando <strong>Streamlit</strong> para la interfaz y <strong>OpenCV</strong> para el procesamiento de imágenes.
        """, unsafe_allow_html=True)

# El bloque 'with st.container():' cierra automáticamente el contenedor