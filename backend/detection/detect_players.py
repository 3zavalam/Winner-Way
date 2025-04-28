from ultralytics import YOLO
import numpy as np

def detect_players_yolo(frames, model_path="backend/models/yolov8_player.pt", conf_threshold=0.3):
    """
    Detecta jugadores en una secuencia de frames usando YOLOv8.

    Args:
        frames (List[np.ndarray]): Lista de frames del video.
        model_path (str): Ruta al modelo YOLOv8 entrenado para jugadores.
        conf_threshold (float): Umbral de confianza para filtrar detecciones.

    Returns:
        List[List[dict]]: Lista por frame con detecciones tipo:
            {"x1": int, "y1": int, "x2": int, "y2": int, "conf": float}
    """
    model = YOLO(model_path)
    detections_per_frame = []

    for frame in frames:
        results = model(frame, verbose=False)[0]

        detections = []
        for box in results.boxes:
            if box.conf[0] < conf_threshold:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            detections.append({
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "conf": conf
            })

        detections_per_frame.append(detections)

    return detections_per_frame
