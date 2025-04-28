import cv2
import json
from pathlib import Path

# CONFIG
METADATA_PATH = "data/metadata.json"
KEYFRAME_OUTPUT_BASE = "data/keyframes/"
MIN_KEYFRAMES_REQUIRED = 2  # puedes subirlo a 3 si quieres asegurar las 3 fases

def load_metadata():
    with open(METADATA_PATH, "r") as f:
        return json.load(f)

def build_keyframe_output_path(output_path):
    return Path(output_path.replace("data/videos/", KEYFRAME_OUTPUT_BASE)).with_suffix(".json")

def draw_label(frame, current_frame, total_frames, label=None):
    text = f"Frame {current_frame + 1}/{total_frames}"
    if label:
        text += f"  ({label})"
    cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def mark_clip(video_path, output_json_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ No se pudo abrir el video: {video_path}")
        return False, False

    keyframes = []
    current_frame = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            print("❌ No se pudo leer el frame.")
            break

        label = next((kf["keyframe_type"] for kf in keyframes if kf["frame_number"] == current_frame), None)
        draw_label(frame, current_frame, total_frames, label)
        cv2.imshow("Mark Keyframes (video)", frame)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('n'):  # siguiente clip
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return True, keyframes  # terminar todo
        elif key == ord('d') and current_frame < total_frames - 1:
            current_frame += 1
        elif key == ord('a') and current_frame > 0:
            current_frame -= 1
        elif key in [ord('p'), ord('i'), ord('f')]:
            label_map = {ord('p'): "preparation", ord('i'): "impact", ord('f'): "follow-through"}
            keyframes = [kf for kf in keyframes if kf["frame_number"] != current_frame]
            keyframes.append({"frame_number": current_frame, "keyframe_type": label_map[key]})
        elif key == ord('c'):
            keyframes = [kf for kf in keyframes if kf["frame_number"] != current_frame]

    cap.release()
    cv2.destroyAllWindows()

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(keyframes, f, indent=2)
    print(f"✅ Guardado: {output_json_path}")
    return False, keyframes

def has_valid_keyframes(keyframe_path):
    try:
        with open(keyframe_path, "r") as f:
            data = json.load(f)
            return len(data) >= MIN_KEYFRAMES_REQUIRED
    except Exception:
        return False

def main():
    metadata = load_metadata()
    print(f"🧠 {len(metadata)} clips en metadata")

    for i, clip in enumerate(metadata):
        output_path = clip["output_path"]
        video_path = Path(output_path)
        keyframe_path = build_keyframe_output_path(output_path)

        if not video_path.exists():
            print(f"⚠️ Clip no encontrado: {video_path}")
            continue

        if keyframe_path.exists() and has_valid_keyframes(keyframe_path):
            print(f"⏩ Ya anotado: {keyframe_path.name} (saltado)")
            continue

        print(f"\n🎬 Clip {i + 1}/{len(metadata)}: {video_path.name}")
        stop_all, _ = mark_clip(video_path, keyframe_path)
        if stop_all:
            break

if __name__ == "__main__":
    main()
