import pandas as pd
import re
import subprocess
from pathlib import Path
from collections import defaultdict
import json

from backend.utils.clipping_dicts import stroke_dict, variant_dict, shot_direction_dict, movement_dict, serve_return_side_dict, serve_target_area_dict, inverted_dict
from backend.utils.name_seconds import clean_name, time_to_seconds

csv_path = "data/clipping.xlsx"
base_dir = Path("data/videos")
input_dir = base_dir / "_raw_videos"

df = pd.read_excel(csv_path)

# Inicializar lista para guardar la metadata
metadata_list = []

counters = {}
errors = []
summary = defaultdict(int)

# Itera sobre las filas del dataframe
for idx, row in df.iterrows():
    try:
        input_video = input_dir / row["input_video"]
        player = str(row['player_name']).replace(" ", "_").lower()
        stroke_type = str(row["stroke_type"]).strip().lower()
        variant = str(row["shot_variant"]).strip().lower()
        hand_style = str(row.get("hand_style", "")).strip().lower()

        # Usar abreviaciones y nombres legibles
        stroke_abbr = stroke_type
        stroke_name = stroke_dict.get(stroke_type, stroke_type)
        stroke_name = clean_name(stroke_name)

        variant_abbr = variant
        variant_name = variant_dict.get(variant, variant)
        variant_name = clean_name(variant_name)

        # Traducir abreviaturas a nombres completos utilizando los diccionarios
        shot_direction_name = shot_direction_dict.get(row["shot_direction"], "no_direction") if not pd.isna(row["shot_direction"]) else "no_direction"
        movement_name = movement_dict.get(row["movement"], "no_movement") if not pd.isna(row["movement"]) else "no_movement"
        serve_return_side_name = serve_return_side_dict.get(row["serve_return_side"], "no_serve_side") if not pd.isna(row["serve_return_side"]) else "no_serve_side"
        serve_target_area_name = serve_target_area_dict.get(row["serve_target_area"], "no_target_area") if not pd.isna(row["serve_target_area"]) else "no_target_area"
        inverted_name = inverted_dict.get(row["inverted"], "False")  # Traducir el valor de True/False a nombre

        # Si stroke_abbr es "bh" y hand_style es uno o dos, agregar el 1H o 2H
        if stroke_abbr == "bh" and hand_style in ["one", "two"]:
            variant_abbr += "1" if hand_style == "one" else "2"
            variant_name += " (1H)" if hand_style == "one" else " (2H)"

        # Contador para generar nombres únicos
        key = f"{player}_{stroke_abbr}_{variant_abbr}"
        counters[key] = counters.get(key, 0) + 1
        number = f"{counters[key]:02d}"

        slowmo = "_slowmo" if str(row["slowmo"]).strip().lower() == "yes" else ""
        output_filename = f"{player}_{stroke_abbr}_{variant_abbr}_{number}{slowmo}.mp4"
       
        # Directorio para metadata
        output_dir = base_dir / stroke_name / variant_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename

        if not input_video.exists():
            raise FileNotFoundError(f"Input file not found: {input_video}")

        # Verify the video duration and adjust if necessary
        start_time = time_to_seconds(row["start_time"])
        duration = float(row["duration"])

        # Crear metadata
        metadata_info = {
            "input_video": row["input_video"],
            "start_time_seconds": start_time,
            "duration_seconds": duration,
            "stroke_type": stroke_name,
            "shot_variant": variant_name,
            "shot_direction": shot_direction_name,
            "movement": movement_name,
            "serve_return_side": serve_return_side_name,
            "serve_target_area": serve_target_area_name,
            "inverted": inverted_name,
            "return": row["return"],
            "player_name": row["player_name"],
            "hand_style": row["hand_style"] if not pd.isna(row["hand_style"]) else "no_hand_style",  # Reemplaza NaN con 'no_hand_style'
            "dominant_hand": row["dominant_hand"],
            "camera_view": row["camera_view"],
            "surface": row["surface"],
            "slowmo": row["slowmo"], 
            "output_filename": output_filename,
            "output_path": str(output_path)  # Agregar la ruta completa del archivo de salida
        }

        # Guardar la metadata en la lista
        metadata_list.append(metadata_info)

        # Actualizar resumen para la metadata procesada
        summary[f"{stroke_name} → {variant_name}"] += 1

    except Exception as e:
        print(f"❌ Error en fila {idx + 2} ({row['input_video']}): {e}")
        errors.append((idx + 2, "N/A", str(e)))

# 🧾 Reporte final
print("\n📊 RESUMEN FINAL:")
total_success = sum(summary.values())
for k, v in summary.items():
    print(f"✔️ {v} metadata entry/entries: {k}")

print(f"\n✅ Total processed: {total_success}")
print(f"❌ Total failed: {len(errors)}")

if errors:
    print("\n⚠️ Errores:")
    for row_num, name, err in errors:
        print(f"  - Fila {row_num} → {name} → {err}")
else:
    print("🎉 ¡Toda la metadata fue procesada correctamente!")

# Guardar metadata en json
with open("data/metadata.json", "w") as json_file:
    json.dump(metadata_list, json_file, indent=4)