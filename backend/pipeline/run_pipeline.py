import argparse
import cv2
import os

from backend.detection.detect_ball import detect_ball_yolo
from backend.detection.detect_players import detect_players_yolo

from backend.tracking.track_players import track_players
from backend.tracking.track_ball import track_ball

from backend.pose.humanpose import HumanPoseExtractor

from backend.utils.draw import draw_ball, draw_players, draw_keypoints, draw_edges
from backend.utils.load_frames import load_video_frames

from backend.utils.normalize_ids import normalize_by_proximity

def run_pipeline(video_path, save_output=False, output_path="data/output/demo_output.mp4"):
    print("📥 Cargando video...")
    frames = load_video_frames(video_path)

    if not frames or frames[0] is None:
        print("❌ Error: No se pudieron cargar los frames del video.")
        return

    print("🎯 Detectando jugadores...")
    player_detections = detect_players_yolo(frames)

    print("🎯 Detectando pelota...")
    ball_detections = detect_ball_yolo(frames)

    print("📌 Trackeando jugadores...")
    tracked_players = track_players(player_detections)
    tracked_players = normalize_by_proximity(tracked_players, target_id=0)

    print("📌 Trackeando pelota...")
    tracked_ball = track_ball(ball_detections)

    pose_extractors = {}

    # Configuración para guardar video
    out = None
    if save_output:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'H264')  # Cambiar a H264
        out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
        if not out.isOpened():
            print("❌ Error: No se pudo inicializar el VideoWriter.")
            return
        print(f"💾 Guardando video en {output_path}")

    print("🖼️ Mostrando resultados (presiona 'q' para salir)...")
    for i in range(len(frames)):
        frame = frames[i].copy()
        draw_players(frame, tracked_players[i])
        draw_ball(frame, tracked_ball[i])

        # Solo tomar el primer jugador (o el más cercano)
        if tracked_players[i]:
            player = tracked_players[i][0]  # Tomamos el primer jugador de la lista
            player_id = player["id"]
            if player_id not in pose_extractors:
                pose_extractors[player_id] = HumanPoseExtractor(frame.shape)
            extractor = pose_extractors[player_id]
            extractor.extract(frame)
            extractor.discard(["left_eye", "right_eye", "left_ear", "right_ear"])
            extractor.roi.update(extractor.keypoints_pixels_frame)

            # Dibujar solo el jugador actual
            draw_keypoints(frame, extractor.keypoints_pixels_frame)
            draw_edges(frame, extractor.keypoints_pixels_frame, extractor.EDGES, 0.2)

        # Guardar el frame procesado en el video
        if out:
            out.write(frame)  # Guardar el frame con keypoints en el video

        cv2.imshow("Detección y Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    if out:
        out.release()
        print(f"✅ Video guardado: {output_path}")

    cv2.destroyAllWindows()
    print("✅ Proceso completado.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Ruta al video de entrada")
    parser.add_argument("--save", type=str, default="false", help="Guardar video renderizado (true/false)")
    parser.add_argument("--output", type=str, default="data/output/demo_output.mp4", help="Ruta para guardar el video")
    args = parser.parse_args()

    save_output = args.save.lower() == "true"
    run_pipeline(args.video, save_output, args.output)