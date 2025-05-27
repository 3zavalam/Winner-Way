import cv2
import mediapipe as mp
import os
import math

mp_pose = mp.solutions.pose

def angle_between(p1, p2, p3):
    a = (p1[0] - p2[0], p1[1] - p2[1])
    b = (p3[0] - p2[0], p3[1] - p2[1])
    dot = a[0]*b[0] + a[1]*b[1]
    norm_a = (a[0]**2 + a[1]**2)**0.5
    norm_b = (b[0]**2 + b[1]**2)**0.5
    if norm_a == 0 or norm_b == 0:
        return 0
    return math.degrees(math.acos(max(min(dot / (norm_a * norm_b), 1.0), -1.0)))

def detect_preparation_frame(video_path, output_path, is_right_handed=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Cannot open video {video_path}")

    os.makedirs(output_path, exist_ok=True)
    pose = mp_pose.Pose(static_image_mode=False)

    best_score = float("inf")
    best_frame = None
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if not results.pose_landmarks or len(results.pose_landmarks.landmark) < 33:
            frame_index += 1
            continue

        lm = results.pose_landmarks.landmark
        if is_right_handed:
            shoulder, elbow, wrist = lm[12], lm[14], lm[16]
        else:
            shoulder, elbow, wrist = lm[11], lm[13], lm[15]

        p_shoulder = (shoulder.x, shoulder.y)
        p_elbow = (elbow.x, elbow.y)
        p_wrist = (wrist.x, wrist.y)

        elbow_angle = angle_between(p_shoulder, p_elbow, p_wrist)
        score = abs(elbow_angle - 110)

        if score < best_score:
            best_score = score
            best_frame = frame.copy()

        frame_index += 1

    cap.release()
    pose.close()

    if best_frame is not None and best_score < 50:  # umbral ajustable
        out_path = os.path.join(output_path, "preparation.jpg")
        cv2.imwrite(out_path, best_frame)
        print(f"Saved preparation frame (score: {best_score:.1f}) to {out_path}")
    else:
        print("No valid preparation frame saved (score too weak or not found).")
