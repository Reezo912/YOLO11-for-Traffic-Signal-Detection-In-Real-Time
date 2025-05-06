from __future__ import annotations
from pathlib import Path
from ultralytics import YOLO
import cv2
from typing import Union, Tuple
from PIL import Image
from file_type import detect_file_type

class SignalDetector:
    """
    Clase para predecir señales en imágenes o videos usando YOLO.

    Args:
        model_path: Ruta al fichero de pesos o nombre del modelo pre-entrenado.
    """

    def __init__(self, model_path: Union[str, Path]) -> None:
        self.model = YOLO(str(model_path))

    def predict_image(self, source: Union[str, Path], imgsz: int, conf: float, iou: float) -> Tuple[bytes, list]:
        results = self.model.predict(source=str(source), show=False, save=False, imgsz=imgsz, conf=conf, iou=iou)
        bgr = results[0].plot()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb, results
    

    def predict_video(
        self, source: Union[str, Path],
        imgsz: int, conf: float, iou: float
    ) -> Tuple[str, list]:
        # Guarda vídeo anotado automáticamente en runs/detect
        results = self.model.predict(
            source=str(source), show=False, save=True,
            imgsz=imgsz, conf=conf, iou=iou
        )
        # results[0].path es la ruta al vídeo anotado
        return results[0].path, results
    
    
    def predict_file(self, source: Union[str, Path], mime: str = None, imgsz: int = 1024, conf: float = 0.5, iou: float = 0.7):
        tipo = detect_file_type(str(source), mime)
        if tipo == "image":
            return self.predict_image(source, imgsz, conf, iou)
        else:
            return self.predict_video(source, imgsz, conf, iou)
        
    
    def predict_frame(
        self,
        frame_bgr: "np.ndarray",
        imgsz: int = 640,
        conf: float = 0.5,
        iou: float = 0.7,
    ) -> "np.ndarray":
        """
        Anota un frame BGR en memoria y lo devuelve en RGB.

        Args:
            frame_bgr: Frame en formato BGR (lo que da OpenCV).
            imgsz, conf, iou:  Parámetros habituales de YOLO.

        Returns:
            annotated_rgb: Frame anotado en RGB listo para st.image().
        """
        results = self.model.predict(
            frame_bgr,            # ahora pasamos directamente el array
            show=False,
            save=False,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
        )
        annotated_bgr = results[0].plot()                 # BGR
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        return annotated_rgb
        
        
    def plot_image(self, source: Union[str, Path], save: bool = False, imgsz: int = 1024, conf: float = 0.5) -> bytes:
        """
        Genera y devuelve la imagen con predicciones para visualizar.

        Args:
            source: Ruta al archivo de imagen.

        Returns:
            imagen_plot: Imagen con detecciones dibujadas (bytes).  
        """
        results = self.model.predict(
            source=str(source),
            save=save,
            imgsz=imgsz,
            conf=conf
        )
        return results[0].plot()

