from scipy.spatial.distance import euclidean

class SimpleBallTracker:
    def __init__(self, max_distance=80, max_missed_frames=10):
        self.next_id = 0
        self.tracks = {}  # {id: {"last_pos": (x, y), "missed": 0, "trajectory": []}}
        self.max_distance = max_distance
        self.max_missed = max_missed_frames

    def update(self, detections):
        assigned = set()
        current_frame = {}

        for track_id, info in self.tracks.items():
            last_pos = info["last_pos"]
            best_det = None
            best_dist = float("inf")

            for i, det in enumerate(detections):
                center = self.get_center(det)
                dist = euclidean(center, last_pos)
                if dist < best_dist and dist < self.max_distance:
                    best_dist = dist
                    best_det = (i, center)

            if best_det:
                i, center = best_det
                assigned.add(i)
                info["last_pos"] = center
                info["trajectory"].append(center)
                info["missed"] = 0
                current_frame[track_id] = {
                    "center": center,
                    "bbox": (det["x1"], det["y1"], det["x2"], det["y2"])
                }

            else:
                info["missed"] += 1

        for i, det in enumerate(detections):
            if i in assigned:
                continue
            center = self.get_center(det)
            self.tracks[self.next_id] = {
                "last_pos": center,
                "missed": 0,
                "trajectory": [center],
            }
            current_frame[self.next_id] = {
                "center": center,
                "bbox": (det["x1"], det["y1"], det["x2"], det["y2"])
            }
            self.next_id += 1

        to_delete = [track_id for track_id, info in self.tracks.items() if info["missed"] > self.max_missed]
        for track_id in to_delete:
            del self.tracks[track_id]

        return current_frame

    def get_center(self, det):
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        return ((x1 + x2) // 2, (y1 + y2) // 2)


def track_ball(detections_per_frame):
    tracker = SimpleBallTracker()
    tracked_frames = []

    for frame_dets in detections_per_frame:
        tracked = tracker.update(frame_dets)
        tracked_frames.append(tracked)  # dict de {id: (x, y)}

    # Contar qué ID apareció más veces
    id_counts = {}
    for frame in tracked_frames:
        for ball_id in frame:
            id_counts[ball_id] = id_counts.get(ball_id, 0) + 1

    if not id_counts:
        print("❌ No se detectó ninguna pelota activa.")
        return [None for _ in detections_per_frame]

    # Elegir el ID que más aparece
    main_id = max(id_counts, key=id_counts.get)
    print(f"✅ Pelota activa seleccionada: ID {main_id} con {id_counts[main_id]} apariciones")

    # Construir trayectoria final de esa pelota
    active_ball_trajectory = []
    for frame in tracked_frames:
        if main_id in frame:
            center = frame[main_id]["center"]
            bbox = frame[main_id]["bbox"]
            active_ball_trajectory.append({
                "x": center[0],
                "y": center[1],
                "x1": bbox[0],
                "y1": bbox[1],
                "x2": bbox[2],
                "y2": bbox[3],
            })
        else:
            active_ball_trajectory.append(None)

    return active_ball_trajectory
