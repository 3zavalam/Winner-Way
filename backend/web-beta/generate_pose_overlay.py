import cv2
import mediapipe as mp
import os
import shutil

mp_pose = mp.solutions.pose

def generate_pose_overlay(input_path, output_path):
    cap = None
    out = None
    pose = None
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video: {input_path}")

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)

        # Usar 'mp4v' para máxima compatibilidad con contenedores .mp4 en navegadores
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
             raise Exception("No se pudo abrir el video para escritura.")

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

        print(f"Video con pose guardado en {output_path}")

    except Exception as e:
        print(f"Error generando el video con pose: {e}. Se usará el video original.")
        # Si algo falla, nos aseguramos de que los recursos se liberen antes de copiar
        if cap: cap.release()
        if out: out.release()
        if pose: pose.close()
        # Copiamos el original como fallback
        shutil.copy(input_path, output_path)
        # Re-asignamos a None para que el bloque finally no intente liberarlos de nuevo
        cap, out, pose = None, None, None

    finally:
        if pose:
            pose.close()
        if cap:
            cap.release()
        if out:
            out.release()