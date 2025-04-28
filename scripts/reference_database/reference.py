import json
from .reference_dicts import reference_federer, reference_alcaraz, reference_djokovic, reference_nadal

# Combinando todos los diccionarios en reference_all
reference_all = {}

# Diccionarios a combinar
reference_dicts = {
    "Federer": reference_federer,
    "Alcaraz": reference_alcaraz,
    "Djokovic": reference_djokovic,
    "Nadal": reference_nadal
}

# Función para combinar los diccionarios
def combine_references(reference_all, player_dicts):
    for player_name, player_dict in player_dicts.items():
        for stroke_type, stroke_dict in player_dict.items():
            if stroke_type not in reference_all:
                reference_all[stroke_type] = {}
            
            for shot_type, reference_id in stroke_dict.items():
                if shot_type not in reference_all[stroke_type]:
                    reference_all[stroke_type][shot_type] = {}
                
                # Añadimos la referencia con el nombre del jugador como clave
                reference_all[stroke_type][shot_type][player_name] = reference_id
    
    return reference_all

# Combinamos todos los diccionarios
reference_all = combine_references(reference_all, reference_dicts)

# Guardamos el resultado en un archivo JSON
output_file = "data/reference_all.json"
with open(output_file, "w") as f:
    json.dump(reference_all, f, indent=4)

print(f"✅ Base de referencia guardada en {output_file}")
