import json
import cv2
import os
from pathlib import Path

from backend.pose.humanpose import HumanPoseExtractor
from backend.utils.normalize_ids import normalize_by_proximity


def export_keypoints_from_metadata(metadata_file="data/metadata.json",
                                   videos_base_path="data/videos/_raw_videos",
                                   output_base_dir="data/json",
                                   model_path="backend/models/movenet.tflite"):
    try:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"❌ Error al leer el metadata: {e}")
        return

    # Agrupar clips por video
    videos_dict = {}
    for clip in metadata:
        video_file = clip["input_video"]
        if video_file not in videos_dict:
            videos_dict[video_file] = []
        videos_dict[video_file].append(clip)

    for video_file, clips in videos_dict.items():
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

        print(f"\n🎬 Procesando: {video_file} ({len(clips)} clips)")
        pose_extractor = HumanPoseExtractor(first_frame.shape, model_path)

        for clip in clips:
            start_frame = int(clip["start_time_seconds"] * fps)
            end_frame = int((clip["start_time_seconds"] + clip["duration_seconds"]) * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            clip_data = []
            frame_in_clip = 0
            current_frame = start_frame

            while current_frame < end_frame:
                ret, frame = cap.read()
                if not ret:
                    print(f"❌ No se pudo leer frame {current_frame} en clip {clip['output_filename']}")
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
                output_path = clip["output_path"].replace("data/videos/", output_base_dir + "/")
                output_json_path = Path(output_path).with_suffix(".json")
                output_json_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_json_path, "w") as f:
                    json.dump(clip_data, f, indent=2)

                print(f"✅ Guardado: {output_json_path}")

        cap.release()


if __name__ == "__main__":
    export_keypoints_from_metadata()