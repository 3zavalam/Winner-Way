import os
import json
import cv2
from pathlib import Path
import sys

# Añadir la raíz del proyecto a sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.pose.humanpose import HumanPoseExtractor
from backend.pipeline.run_pipeline import run_pipeline

def export_keypoints_for_reddit(metadata_file="data/metadata_reddit.json",
                                videos_base_path="data/videos/reddit_clips",
                                output_base_dir="data/json/reddit_clips",
                                model_path="backend/models/movenet.tflite"):
    try:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"❌ Error al leer el metadata: {e}")
        return

    for clip in metadata:
        video_file = clip["clip_name"]
        video_path = Path(videos_base_path) / video_file

        if not video_path.exists():
            print(f"⚠️ Video no encontrado: {video_path}")
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ No se pudo abrir el video: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        ret, first_frame = cap.read()
        if not ret:
            print(f"❌ No se pudo leer el primer frame del video {video_path}")
            cap.release()
            continue

        print(f"\n🎬 Procesando: {video_file}")
        pose_extractor = HumanPoseExtractor(first_frame.shape, model_path)

        start_frame = int(clip["start_time"] * fps)
        end_frame = int((clip["end_time"]) * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        clip_data = []
        frame_in_clip = 0
        current_frame = start_frame

        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                print(f"❌ No se pudo leer frame {current_frame} en clip {clip['clip_name']}")
                break

            pose_extractor.extract(frame)
            pose_extractor.discard(["left_eye", "right_eye", "left_ear", "right_ear"])
            pose_extractor.roi.update(pose_extractor.keypoints_pixels_frame)

            frame_data = {
                "frame_number": current_frame,
                "frame_in_clip": frame_in_clip,
                "timestamp": round(current_frame / fps, 3),
                "keypoints": pose_extractor.get_keypoints_list(),
                "roi": {
                    "center_x": pose_extractor.roi.center_x,
                    "center_y": pose_extractor.roi.center_y,
                    "width": pose_extractor.roi.width,
                    "height": pose_extractor.roi.height,
                    "valid": pose_extractor.roi.valid
                },
                "stroke_type": clip.get("stroke_type"),
                "player_name": clip.get("player_name"),
                "camera_view": clip.get("camera_view"),
                "output_filename": clip.get("output_filename")
            }

            clip_data.append(frame_data)
            frame_in_clip += 1
            current_frame += 1

        # Guardar archivo JSON
        if clip_data:
            output_path = clip["output_filename"].replace("data/videos/", output_base_dir + "/")
            output_json_path = Path(output_path).with_suffix(".json")
            output_json_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_json_path, "w") as f:
                json.dump(clip_data, f, indent=2)

            print(f"✅ Guardado: {output_json_path}")

        cap.release()


def run_full_pipeline(video_path, output_video_path=f"data/output/demo_output.mp4"):
    # Ejecuta el pipeline completo de detección, tracking, normalización y visualización
    run_pipeline(
        video_path=video_path,
        save_output=True
    )


def main():
    # Paso 1: Extraer keypoints para los clips del archivo metadata
    export_keypoints_for_reddit()

    # Paso 2: Ejecutar el pipeline completo para un clip
    video_clip_path = "data/videos/reddit_clips/Panos217_serve.mp4"  
    run_full_pipeline(video_clip_path)


if __name__ == "__main__":
    main()