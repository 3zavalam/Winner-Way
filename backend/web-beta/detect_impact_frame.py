import cv2
import mediapipe as mp
import os
import math

mp_pose = mp.solutions.pose

def angle_between(p1, p2, p3):
    a = (p1[0] - p2[0], p1[1] - p2[1])
    b = (p3[0] - p2[0], p3[1] - p2[1])
    dot = a[0]*b[0] + a[1]*b[1]
    norm_a = math.hypot(*a)
    norm_b = math.hypot(*b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cos_theta = max(min(dot / (norm_a * norm_b), 1.0), -1.0)
    return math.degrees(math.acos(cos_theta))

mp_pose = mp.solutions.pose

def detect_impact_frame(video_path, output_path, is_right_handed=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Cannot open video {video_path}")

    os.makedirs(output_path, exist_ok=True)
    

    pose = mp_pose.Pose(static_image_mode=False)
    frames = []
    scores = []
    best_index = -1
    max_score = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Si el jugador es zurdo, espejamos el frame para usar siempre la misma referencia
        if not is_right_handed:
            frame = cv2.flip(frame, 1)

        frames.append(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if not results.pose_landmarks:
            scores.append(-1)
            continue

        lm = results.pose_landmarks.landmark
        # Selección de hombro y muñeca según mano dominante
        if is_right_handed:
            shoulder = lm[12]
            wrist    = lm[16]
        else:
            shoulder = lm[11]
            wrist    = lm[15]

        dx = wrist.x - shoulder.x
        dy = wrist.y - shoulder.y
        dist = math.hypot(dx, dy)
        scores.append(dist)

        if dist > max_score:
            max_score  = dist
            best_index = len(frames) - 1

    cap.release()
    pose.close()

    # Fallback: buscar ±5 frames alrededor con pose válida
    impact_frame = None
    for offset in range(-5, 6):
        idx = best_index + offset
        if 0 <= idx < len(frames):
            rgb = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
            with mp_pose.Pose(static_image_mode=True) as fallback_pose:
                result = fallback_pose.process(rgb)
                if result.pose_landmarks and len(result.pose_landmarks.landmark) == 33:
                    impact_frame = frames[idx]
                    break

    if impact_frame is not None:
        out_path = os.path.join(output_path, "impact.jpg")
        cv2.imwrite(out_path, impact_frame)
        print(f"✅ Saved impact frame (with fallback) to {out_path}")
    else:
        print("❌ No valid impact frame found.")