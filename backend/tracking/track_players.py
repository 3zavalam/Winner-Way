import numpy as np
from scipy.spatial.distance import euclidean

class SimplePlayerTracker:
    def __init__(self, max_distance=100, max_missed_frames=5):
        self.next_id = 0
        self.tracks = {}  # {id: {"last_pos": (x, y), "missed": 0}}
        self.max_distance = max_distance
        self.max_missed = max_missed_frames

    def update(self, detections):
        assigned = set()
        current_frame = []

        # Paso 1: Asignar detecciones a tracks existentes
        for track_id, info in self.tracks.items():
            last_pos = info["last_pos"]
            best_det = None
            best_dist = float("inf")

            for i, det in enumerate(detections):
                center = self.get_center(det)
                dist = euclidean(center, last_pos)
                if dist < best_dist and dist < self.max_distance:
                    best_dist = dist
                    best_det = (i, center, det)

            if best_det:
                i, center, det = best_det
                assigned.add(i)
                info["last_pos"] = center
                info["missed"] = 0
                det["id"] = track_id
                current_frame.append(det)
            else:
                info["missed"] += 1

        # Paso 2: Crear nuevos tracks para detecciones no asignadas
        for i, det in enumerate(detections):
            if i in assigned:
                continue
            center = self.get_center(det)
            self.tracks[self.next_id] = {
                "last_pos": center,
                "missed": 0,
            }
            det["id"] = self.next_id
            current_frame.append(det)
            self.next_id += 1

        # Paso 3: Eliminar tracks que se perdieron
        to_delete = [track_id for track_id, info in self.tracks.items() if info["missed"] > self.max_missed]
        for track_id in to_delete:
            del self.tracks[track_id]

        return current_frame

    def get_center(self, det):
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        return ((x1 + x2) // 2, (y1 + y2) // 2)


def track_players(detections_per_frame):
    tracker = SimplePlayerTracker()
    tracked_per_frame = []

    for frame_dets in detections_per_frame:
        tracked = tracker.update(frame_dets)
        tracked_per_frame.append(tracked)

    return tracked_per_frame

def get_closest_id_to_camera(detections, frame_height):
    max_y = -1
    main_id = None
    for det in detections:
        y = det["y2"]  # parte baja de la caja
        if y > max_y:
            max_y = y
            main_id = det["id"]
    return main_id
