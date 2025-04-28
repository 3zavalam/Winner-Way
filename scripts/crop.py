import pandas as pd
import subprocess
from pathlib import Path
from collections import defaultdict

from backend.utils.clipping_dicts import stroke_dict, variant_dict
from backend.utils.name_seconds import clean_name, time_to_seconds

# Define paths
csv_path = "data/clipping.xlsx"
base_dir = Path("data/videos")
input_dir = base_dir / "_raw_videos"

# Read the CSV file
df = pd.read_excel(csv_path)

# Initialize counters and summary
counters = {}
errors = []
summary = defaultdict(int)

# Iterate over rows of the dataframe
for idx, row in df.iterrows():
    try:
        input_video = input_dir / row["input_video"]
        player = str(row['player_name']).replace(" ", "_").lower()
        stroke_type = str(row["stroke_type"]).strip().lower()
        variant = str(row["shot_variant"]).strip().lower()
        hand_style = str(row.get("hand_style", "")).strip().lower()

        # Use abbreviations and readable names
        stroke_abbr = stroke_type
        stroke_name = stroke_dict.get(stroke_type, stroke_type)
        stroke_name = clean_name(stroke_name)

        variant_abbr = variant
        variant_name = variant_dict.get(variant, variant)
        variant_name = clean_name(variant_name)

        if stroke_abbr == "bh" and hand_style in ["one", "two"]:
            variant_abbr += "1" if hand_style == "one" else "2"
            variant_name += " (1H)" if hand_style == "one" else " (2H)"

        key = f"{player}_{stroke_abbr}_{variant_abbr}"
        counters[key] = counters.get(key, 0) + 1
        number = f"{counters[key]:02d}"

        slowmo = "_slowmo" if str(row["slowmo"]).strip().lower() == "yes" else ""
        output_filename = f"{player}_{stroke_abbr}_{variant_abbr}_{number}{slowmo}.mp4"
       
        # Determine output folder based on 'return' value
        if row.get('return', False):  # If 'return' is TRUE
            output_dir = base_dir / stroke_name / "return" / variant_name
        else:
            output_dir = base_dir / stroke_name / variant_name
        
        # Create the output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename

        # Ensure the input file exists
        if not input_video.exists():
            raise FileNotFoundError(f"Input file not found: {input_video}")

        # Verify the video duration and adjust if necessary
        start_time = time_to_seconds(row["start_time"])
        duration = float(row["duration"])

        # Construct ffmpeg command for clipping
        cmd = [
            "ffmpeg",
            "-i", str(input_video),
            "-ss", str(start_time), 
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "ultrafast",  # Fast encoding preset
            "-crf", "28",  # Adjust CRF for speed/quality balance
            "-pix_fmt", "yuv420p",  # Ensure pixel format is compatible
            str(output_path)
        ]

        # Print the command being executed
        print(f"🎾 Recortando: {output_filename}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Handle ffmpeg errors
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error for {output_filename}: {result.stderr}")

        summary[f"{stroke_name} → {variant_name}"] += 1

    except Exception as e:
        print(f"❌ Error en fila {idx + 2} ({row['input_video']}): {e}")
        # Ensure output_filename is defined before adding the error
        if 'output_filename' in locals():
            errors.append((idx + 2, output_filename, str(e)))
        else:
            errors.append((idx + 2, "N/A", str(e)))

# 🧾 Final report
print("\n📊 RESUMEN FINAL:")
total_success = sum(summary.values())
for k, v in summary.items():
    print(f"✔️ {v} clip(s): {k}")

print(f"\n✅ Total generados: {total_success}")
print(f"❌ Total fallidos: {len(errors)}")

if errors:
    print("\n⚠️ Errores:")
    for row_num, name, err in errors:
        print(f"  - Fila {row_num} → {name} → {err}")
else:
    print("🎉 ¡Todos los clips fueron generados sin errores!")