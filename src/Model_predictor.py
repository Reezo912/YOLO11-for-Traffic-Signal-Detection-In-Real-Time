from pathlib import Path
from typing import Union, Tuple, List, Optional
import cv2
from ultralytics import YOLO
from file_type import detect_file_type  # compruba si el archivo es una imagen o video
import torch as torch

torch.set_num_threads(1)          # evita picos de RAM


class SignalDetector:
    """
    Detector de señales con YOLO.
    Args:
        model_path: ruta a pesos o modelo
        lista_ignorar_clases: lista de índices a ignorar, o None para no filtrar
    """

    def __init__(self, model_path: Union[str, Path], device='cuda:0',lista_ignorar_clases: Optional[List[int]] = None) -> None:
        self.model = YOLO(str(model_path))
        #  ⬇️ fuerza GPU si existe, si no se queda en CPU
        if device.startswith("cuda") and torch.cuda.is_available():
            self.model.to(device)
            self.model.fuse()          # opcional: fusiona capas Conv+BN
            self.model.half()          # FP16 → más rápido, menos VRAM


        if lista_ignorar_clases is None:
            # None no pasa el filtro de clase
            self.clases_a_mantener = None
        else:
            # se pasan todas las clases menos las que quiero ignorar
            self.clases_a_mantener = [idx for idx in self.model.names.keys() if idx not in lista_ignorar_clases]


    def _predict_with_filter(self, *args, **predict_kwargs):
        # Se pasa la lista de clases en caso de haberla
        if self.clases_a_mantener is not None:
            predict_kwargs['classes'] = self.clases_a_mantener
        return self.model.predict(*args, **predict_kwargs)


    def predict_image(self, source: Union[str, Path], imgsz: int, conf: float, iou: float) -> Tuple["np.ndarray", list]:
        results = self._predict_with_filter(source=str(source), show=False, save=False, imgsz=imgsz, conf=conf, iou=iou)
        bgr = results[0].plot()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb, results


    def predict_video(self, source: Union[str, Path], imgsz: int, conf: float, iou: float) -> Tuple[str, list]:
        results = self._predict_with_filter(source=str(source), show=False, save=True, imgsz=imgsz, conf=conf, iou=iou)
        return results[0].path, results


    def predict_file(self, source: Union[str, Path], mime: str = None, imgsz: int = 1024, conf: float = 0.5, iou: float = 0.7):
        tipo = detect_file_type(str(source), mime)
        if tipo == "image":
            return self.predict_image(source, imgsz, conf, iou)
        else:
            return self.predict_video(source, imgsz, conf, iou)
        

    def predict_frame(self, frame_bgr: "np.ndarray", imgsz: int = 640, conf: float = 0.5, iou: float = 0.7) -> "np.ndarray":
        results = self._predict_with_filter(frame_bgr, show=False, save=False, imgsz=imgsz, conf=conf, iou=iou)
        annotated_bgr = results[0].plot()
        return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), results


    def plot_image(self, source: Union[str, Path], save: bool = False, imgsz: int = 1024, conf: float = 0.5) -> "np.ndarray":
        results = self._predict_with_filter(source=str(source), save=save, imgsz=imgsz, conf=conf)
        return results[0].plot()




if __name__ == '__main__':

    detector = SignalDetector('./models/SignalDetector.pt')

    # después de detector = SignalDetector(...)
    import torch, inspect
    p = next(detector.model.model.parameters())
    print("⮕  Weights device:", p.device)          # debe decir cuda:0
    print("⮕  dtype         :", p.dtype)           # debe decir torch.float16
