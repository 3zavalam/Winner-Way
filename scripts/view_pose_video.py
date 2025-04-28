import json
import cv2
import numpy as np
import time
from pathlib import Path

EDGES = {
    (0, 1): (255, 0, 255), (0, 2): (0, 255, 255), (1, 3): (255, 0, 255), (2, 4): (0, 255, 255),
    (0, 5): (255, 0, 255), (0, 6): (0, 255, 255), (5, 7): (255, 0, 255), (7, 9): (255, 0, 255),
    (6, 8): (0, 255, 255), (8, 10): (0, 255, 255), (5, 6): (255, 255, 0), (5, 11): (255, 0, 255),
    (6, 12): (0, 255, 255), (11, 12): (255, 255, 0), (11, 13): (255, 0, 255), (13, 15): (255, 0, 255),
    (12, 14): (0, 255, 255), (14, 16): (0, 255, 255)
}

def draw_pose(frame, keypoints):
    for kp in keypoints:
        x, y, conf = int(kp['x']), int(kp['y']), kp['confidence']
        if conf > 0.1:
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    for (i, j), color in EDGES.items():
        if keypoints[i]['confidence'] > 0.1 and keypoints[j]['confidence'] > 0.1:
            x1, y1 = int(keypoints[i]['x']), int(keypoints[i]['y'])
            x2, y2 = int(keypoints[j]['x']), int(keypoints[j]['y'])
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)

def view_pose_video(json_path, size=(1280, 720), fps=30):
    with open(json_path) as f:
        data = json.load(f)

    for i, frame_data in enumerate(data):
        frame = np.ones((size[1], size[0], 3), dtype=np.uint8) * 20  # fondo oscuro
        draw_pose(frame, frame_data["keypoints"])

        cv2.putText(frame, f"Frame: {i+1}/{len(data)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)

        cv2.imshow("Pose Video", frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_json_path = "/Users/emilio/Documents/Winner Way/data/json/Backhand/Drop_Shot (1H)/roger_federer_bh_ds1_01.json"
    view_pose_video(video_json_path)