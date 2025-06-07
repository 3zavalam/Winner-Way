import cv2
import mediapipe as mp
import os

mp_pose = mp.solutions.pose

def generate_pose_overlay(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el video de entrada: {input_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # Cambiado a 'mp4v' por ser más compatible con servidores Linux que 'avc1' (H.264)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        cap.release()
        raise IOError(f"No se pudo crear el video de salida en {output_path}. Revisa los códecs.")

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Optimización: Marcar el frame como no escribible para MediaPipe
        frame.flags.writeable = False
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = pose.process(rgb_frame)
        
        # Volver a permitir escritura para dibujar los landmarks
        frame.flags.writeable = True

        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

        out.write(frame)

    cap.release()
    out.release()
    pose.close()
    print(f"Overlay de pose guardado en {output_path}")