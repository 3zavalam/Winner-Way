import json
from pathlib import Path

# Rutas base
METADATA_PATH = "data/metadata.json"
KEYPOINTS_BASE = "data/json/"
KEYFRAMES_BASE = "data/keyframes/"
OUTPUT_SUFFIX = "_keypoints.json"

def build_keypoints_path(output_path):
    return Path(output_path.replace("data/videos/", KEYPOINTS_BASE)).with_suffix(".json")

def build_keyframes_path(output_path):
    return Path(output_path.replace("data/videos/", KEYFRAMES_BASE)).with_suffix(".json")

def build_output_path(output_path):
    path = Path(output_path.replace("data/videos/", KEYPOINTS_BASE))
    return path.with_name(path.stem + OUTPUT_SUFFIX)

def annotate_clip(keypoints_path, keyframes_path, output_path):
    if not keypoints_path.exists():
        print(f"❌ No se encontró keypoints: {keypoints_path}")
        return
    if not keyframes_path.exists():
        print(f"⚠️  No hay keyframes para: {keypoints_path.name} (omitido)")
        return

    with open(keypoints_path) as f:
        keypoints_data = json.load(f)

    with open(keyframes_path) as f:
        keyframes = json.load(f)

    # 🔁 Mapeamos usando frame_in_clip
    keyframe_map = {kf["frame_number"]: kf["keyframe_type"] for kf in keyframes}

    annotated_count = 0
    for frame in keypoints_data:
        frame_index = frame.get("frame_in_clip")
        if frame_index in keyframe_map:
            frame["keyframe_type"] = keyframe_map[frame_index]
            annotated_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(keypoints_data, f, indent=2)

    print(f"✅ Guardado {output_path.name}  [{annotated_count} keyframes anotados]")

def main():
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    for clip in metadata:
        output_path = clip["output_path"]

        keypoints_path = build_keypoints_path(output_path)
        keyframes_path = build_keyframes_path(output_path)
        output_annotated_path = build_output_path(output_path)

        annotate_clip(keypoints_path, keyframes_path, output_annotated_path)

if __name__ == "__main__":
    main()