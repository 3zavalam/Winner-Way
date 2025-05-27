import cv2
import mediapipe as mp
import os

mp_pose = mp.solutions.pose

def generate_pose_overlay(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise Exception("Could not open video")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # Cambiar el codec a H.264
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # alternativamente "H264"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    pose = mp_pose.Pose()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        out.write(frame)

    cap.release()
    out.release()
    pose.close()
    print(f"Saved pose overlay video to {output_path}")