import matplotlib.pyplot as plt
from pathlib import Path

from compare_clips import compare_with_dtw

clips_json = 'data/reference_all.json'

# Función para comparar varios clips y graficar la distancia DTW
def compare_multiple_clips(reference_clip, clips_to_compare):
    distances = []
    for clip in clips_to_compare:
        distance = compare_with_dtw(reference_clip, clip)
        distances.append(distance)
    
    # Graficar las distancias
    plt.bar(range(len(clips_to_compare)), distances)
    plt.xlabel('Clips')
    plt.ylabel('DTW Distance')
    plt.title('Comparación de Distancia DTW')
    plt.xticks(range(len(clips_to_compare)), [Path(clip).stem for clip in clips_to_compare], rotation=90)
    plt.show()

# Ejemplo de uso
def main():
    full = "/Users/emilio/Documents/Winner Way/"
    reference_clip = f"{full}data/json/Forehand/Topspin/roger_federer_fh_ts_02_keypoints.json"
    clips_to_compare = [
        f"{full}data/json/Forehand/Topspin/carlos_alcaraz_fh_ts_04_keypoints.json",
        f"{full}data/json/Forehand/Topspin/novak_djokovic_fh_ts_02_keypoints.json",
    ]
    
    compare_multiple_clips(reference_clip, clips_to_compare)

if __name__ == "__main__":
    main()
