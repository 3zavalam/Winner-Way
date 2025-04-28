import cv2

def draw_keypoints(frame, keypoints, confidence_threshold=0.1):
    """Dibuja los keypoints detectados en el frame"""
    for i, (y, x, confidence) in enumerate(keypoints):
        if confidence > confidence_threshold:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Dibujar círculo en cada keypoint

def draw_edges(frame, shaped, edges, confidence_threshold):
    """Dibuja las conexiones entre keypoints con colores personalizados"""
    COLORS = {"c": (255, 255, 0), "m": (255, 0, 255), "y": (0, 255, 255)}  # local

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            color_to_use = COLORS.get(color, (0, 255, 255))
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color=color_to_use, thickness=2)

def draw_roi(roi, frame):
    """Dibuja la Región de Interés (RoI) con un cuadro amarillo"""
    # Dibujar las 4 líneas que forman el cuadrado de la ROI
    cv2.line(frame, 
             (roi.center_x - roi.width // 2, roi.center_y - roi.height // 2),
             (roi.center_x - roi.width // 2, roi.center_y + roi.height // 2),
             (0, 255, 255), 3)  # Línea izquierda (amarillo)

    cv2.line(frame, 
             (roi.center_x + roi.width // 2, roi.center_y + roi.height // 2),
             (roi.center_x - roi.width // 2, roi.center_y + roi.height // 2),
             (0, 255, 255), 3)  # Línea inferior (amarillo)

    cv2.line(frame, 
             (roi.center_x + roi.width // 2, roi.center_y + roi.height // 2),
             (roi.center_x + roi.width // 2, roi.center_y - roi.height // 2),
             (0, 255, 255), 3)  # Línea derecha (amarillo)

    cv2.line(frame, 
             (roi.center_x - roi.width // 2, roi.center_y - roi.height // 2),
             (roi.center_x + roi.width // 2, roi.center_y - roi.height // 2),
             (0, 255, 255), 3)  # Línea superior (amarillo)
    
"""def draw_players(frame, player_dets, color=(0, 255, 0)):
    for det in player_dets:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        player_id = det["id"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"P{player_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)"""
def draw_players(frame, player_dets, color=(0, 255, 0)):
    for det in player_dets:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        player_id = det["id"]  # asegúrate que sea esto
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"P{player_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



def draw_ball(frame, ball_point):
    if not ball_point or "x" not in ball_point:
        return

    x, y = ball_point["x"], ball_point["y"]
    x1, y1, x2, y2 = ball_point["x1"], ball_point["y1"], ball_point["x2"], ball_point["y2"]
    cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(frame, "Ball", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
