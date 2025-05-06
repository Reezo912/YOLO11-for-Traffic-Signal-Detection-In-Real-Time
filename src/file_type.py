import os

def detect_file_type(path: str, mime: str = None) -> str:
    """
    Detecta si el archivo es imagen o vídeo.
    """
    if mime:
        if mime.startswith("image/"):
            return "image"
        if mime.startswith("video/"):
            return "video"
    ext = os.path.splitext(path)[1].lower()
    if ext in {".jpg", ".jpeg", ".png", ".bmp", ".gif"}:
        return "image"
    if ext in {".mp4", ".avi", ".mov", ".mkv"}:
        return "video"
    raise ValueError(f"Formato no soportado: {mime or ext}")