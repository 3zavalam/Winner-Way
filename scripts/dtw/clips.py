import numpy as np
import json
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from dtaidistance import dtw
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from pathlib import Path
from typing import List, Optional, Union
import logging

class TennisMotionComparator:
    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        """
        Inicializa el comparador de movimientos de tenis.
        
        Args:
            base_dir: Directorio base del proyecto (opcional). 
                      Si no se provee, busca hacia arriba desde este archivo 
                      hasta encontrar una carpeta 'data'.
        """
        # Configurar logger
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

        # Determina raíz de proyecto
        if base_dir:
            root = Path(base_dir).resolve()
        else:
            # Empieza en el directorio de este script
            root = Path(__file__).resolve().parent

        # Busca hacia arriba la carpeta que contenga 'data'
        for _ in range(5):
            if (root / "data").is_dir():
                break
            root = root.parent
        else:
            raise FileNotFoundError(f"No se encontró carpeta 'data' en ningún ancestro de {Path(base_dir) if base_dir else __file__}")

        self.base_dir = root
        self.data_dir = self.base_dir / "data"
        # Si existe subcarpeta 'json', la usa; si no, asume que los .json están directamente en data/
        cand_json = self.data_dir / "json"
        self.json_dir = cand_json if cand_json.is_dir() else self.data_dir

        logging.info(f"Directorio de datos: {self.data_dir}")
        logging.info(f"Directorio de JSON:  {self.json_dir}")

        self.reference_data = {}
        self.keypoints_data = {}
        self.shot_metadata  = {}
        self.keypoint_names  = [
            "nose", "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow", "left_wrist", "right_wrist",
            "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle"
        ]

    def load_reference_file(self, reference_file: str = "reference_all.json") -> bool:
        """
        Carga el archivo de referencia que contiene la clasificación de golpes.
        
        Args:
            reference_file: Nombre del JSON (con o sin ".json")
        
        Returns:
            True si se cargó correctamente, False en caso contrario.
        """
        fname = reference_file if reference_file.endswith(".json") else reference_file + ".json"
        path  = self.data_dir / fname

        try:
            with path.open("r", encoding="utf-8") as f:
                self.reference_data = json.load(f)
            logging.info(f"✔ Referencia cargada: {path}")
            return True

        except FileNotFoundError:
            logging.error(f"✖ Referencia no encontrada: {path}")
            return False
        except json.JSONDecodeError as e:
            logging.error(f"✖ JSON mal formado en {path}: {e}")
            return False
        except Exception as e:
            logging.error(f"✖ Error al cargar referencia {path}: {e}")
            return False

    def find_keypoints_file(self, shot_id: str) -> Optional[Path]:
        """
        Busca el archivo *_keypoints.json correspondiente a un shot_id.
        
        Args:
            shot_id: Identificador del golpe (sin sufijo)
        
        Returns:
            Path al archivo si existe, o None.
        """
        pattern = f"{shot_id}_keypoints.json"
        for p in self.json_dir.rglob(pattern):
            return p
        return None

    def load_keypoints_file(self, shot_id: str) -> bool:
        """
        Carga un archivo de keypoints específico y extrae metadata.
        
        Args:
            shot_id: Identificador del golpe (sin sufijo)
        
        Returns:
            True si se cargó correctamente, False en caso contrario.
        """
        file_path = self.find_keypoints_file(shot_id)
        if not file_path:
            logging.error(f"No se encontró el archivo de keypoints para: {shot_id}")
            return False

        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            self.keypoints_data[shot_id] = data

            # Extraer stroke_type y spin_type de la ruta
            parts = file_path.parts
            stroke_type = next((p for p in parts if p in ["Backhand", "Forehand", "Serve", "Smash"]), None)
            spin_type = None
            if stroke_type:
                idx = parts.index(stroke_type)
                if idx + 1 < len(parts) and parts[idx+1] not in {"data", "json"}:
                    spin_type = parts[idx+1]

            # Extraer player a partir del shot_id
            tokens = shot_id.split("_")
            player = f"{tokens[0].capitalize()} {tokens[1].capitalize()}" if len(tokens) >= 2 else None

            # Guardar metadata
            self.shot_metadata[shot_id] = {
                "stroke_type": stroke_type,
                "spin_type": spin_type,
                "player": player,
                "file_path": str(file_path)
            }

            logging.info(f"✔ Keypoints cargados: {shot_id}")
            return True

        except Exception as e:
            logging.error(f"✖ Error al cargar {shot_id} desde {file_path}: {e}")
            return False

    def load_all_reference_shots(self) -> int:
        """
        Carga todos los golpes listados en el archivo de referencia.

        Returns:
            Número de archivos de keypoints cargados con éxito.
        """
        if not self.reference_data:
            logging.error("No hay datos de referencia. Ejecuta primero load_reference_file().")
            return 0

        count = 0
        for stroke_type, spins in self.reference_data.items():
            for spin_type, players in spins.items():
                for player_name, shot_id in players.items():
                    if self.load_keypoints_file(shot_id):
                        count += 1

        logging.info(f"✔ Total de keypoints de referencia cargados: {count}")
        return count

    def load_shots_by_category(
        self,
        stroke_type: Optional[str] = None,
        spin_type:   Optional[str] = None,
        player:      Optional[str] = None
    ) -> List[str]:
        """
        Carga los golpes que coinciden con los filtros dados.

        Args:
            stroke_type: Filtra por tipo de golpe.
            spin_type:   Filtra por tipo de efecto.
            player:      Filtra por nombre de jugador (substring).

        Returns:
            Lista de shot_id cargados.
        """
        if not self.reference_data:
            logging.error("No hay datos de referencia. Ejecuta primero load_reference_file().")
            return []

        loaded: List[str] = []
        for s_type, spins in self.reference_data.items():
            if stroke_type and s_type != stroke_type:
                continue
            for s_spin, players in spins.items():
                if spin_type and s_spin != spin_type:
                    continue
                for p_name, shot_id in players.items():
                    if player and player.lower() not in p_name.lower():
                        continue
                    if shot_id not in loaded and self.load_keypoints_file(shot_id):
                        loaded.append(shot_id)

        logging.info(f"✔ Keypoints cargados por categoría: {len(loaded)}")
        return loaded
    
    def load_all_available_shots(self):
        """
        Carga todos los archivos de keypoints disponibles en la estructura de directorios.
        
        Returns:
            Número de archivos cargados
        """
        loaded_count = 0
        
        # Recorrer toda la estructura de directorios
        for root, _, files in os.walk(self.json_dir):
            for file in files:
                if file.endswith('_keypoints.json'):
                    shot_id = file.replace('_keypoints.json', '')
                    if self.load_keypoints_file(shot_id):
                        loaded_count += 1
        
        print(f"Se cargaron {loaded_count} archivos de keypoints disponibles.")
        return loaded_count
    
    def extract_pose_sequence(self, shot_id):
        """
        Extrae la secuencia de poses de un golpe como una matriz numpy.
        
        Args:
            shot_id: Identificador del golpe
            
        Returns:
            Matriz numpy de forma (n_frames, n_keypoints*2) con coordenadas x,y concatenadas
        """
        if shot_id not in self.keypoints_data:
            print(f"El golpe {shot_id} no está cargado")
            return None
        
        clip_data = self.keypoints_data[shot_id]
        frames = []
        
        # Ordenamos los frames por número de frame
        sorted_frames = sorted(clip_data, key=lambda x: x.get('frame_number', 0) if isinstance(x, dict) else 0)
        
        for frame_data in sorted_frames:
            # Asegurarse de que tenemos los keypoints
            if not isinstance(frame_data, dict) or 'keypoints' not in frame_data:
                continue
            
            # Extraer las coordenadas x, y de cada keypoint
            keypoints = frame_data['keypoints']
            if not keypoints:
                continue
            
            # Crear un vector de características para este frame
            frame_features = []
            for kp in keypoints:
                # Solo usamos keypoints con confianza mayor a un umbral
                if kp.get('confidence', 0) > 0.2:
                    frame_features.extend([kp.get('x', 0), kp.get('y', 0)])
                else:
                    # Si la confianza es baja, usamos valores nulos
                    frame_features.extend([np.nan, np.nan])
            
            # Añadir este frame si tiene suficientes características
            if len(frame_features) > 0:
                frames.append(frame_features)
        
        # Convertir a numpy array
        if frames:
            sequence = np.array(frames)
            # Manejar valores NaN interpolando
            if np.isnan(sequence).any():
                # Interpolación simple para valores faltantes
                df = pd.DataFrame(sequence)
                df = df.interpolate(method='linear', limit_direction='both')
                sequence = df.to_numpy()
            return sequence
        else:
            return np.array([])
    
    def normalize_sequence(self, sequence):
        """
        Normaliza una secuencia de poses para hacerla invariante a la escala y posición.
        
        Args:
            sequence: Matriz numpy de forma (n_frames, n_features)
            
        Returns:
            Secuencia normalizada
        """
        if len(sequence) == 0:
            return sequence
        
        normalized = sequence.copy()
        
        # Para cada frame
        for i in range(len(normalized)):
            frame = normalized[i].reshape(-1, 2)  # Reshape a pares (x,y)
            
            # Calcular el centro (centroide) de la pose
            valid_points = ~np.isnan(frame).any(axis=1)
            if sum(valid_points) > 0:
                center = np.mean(frame[valid_points], axis=0)
                
                # Restar el centro para centrar la pose
                frame = frame - center
                
                # Normalizar por la distancia máxima al centro
                max_dist = np.max(np.linalg.norm(frame[valid_points], axis=1))
                if max_dist > 0:
                    frame = frame / max_dist
                
                # Volver a aplanar
                normalized[i] = frame.flatten()
        
        return normalized
    
    def calculate_dtw_distance(self, seq1, seq2, method='fastdtw'):
        """
        Calcula la distancia DTW entre dos secuencias, limpiando NaNs e Infs.

        Args:
            seq1, seq2: Secuencias de poses (listas o arrays) normalizadas.
            method:     Método DTW a usar ('fastdtw' o 'dtaidistance').

        Returns:
            Tuple[float, list]: (distancia DTW, camino de alineación)
        """
        # Si una secuencia está vacía, distancia infinita
        if len(seq1) == 0 or len(seq2) == 0:
            return float('inf'), []

        # Asegurar array NumPy de tipo float
        arr1 = np.asarray(seq1, dtype=float)
        arr2 = np.asarray(seq2, dtype=float)

        # Reemplazar NaN y ±Inf por 0.0 (o podrías usar otro valor/neutro)
        arr1 = np.nan_to_num(arr1, nan=0.0, posinf=0.0, neginf=0.0)
        arr2 = np.nan_to_num(arr2, nan=0.0, posinf=0.0, neginf=0.0)

        if method == 'fastdtw':
            distance, path = fastdtw(arr1, arr2, dist=euclidean)
            return distance, path

        elif method == 'dtaidistance':
            distance = dtw.distance_fast(arr1, arr2)
            # dtaidistance no devuelve camino por defecto
            return distance, []

        else:
            raise ValueError(f"Método DTW no reconocido: {method}")
    
    def compare_shots(self, shot_id1, shot_id2, normalize=True, method='fastdtw'):
        """
        Compara dos golpes específicos usando DTW.
        
        Args:
            shot_id1, shot_id2: Identificadores de los golpes a comparar
            normalize: Si se normalizan las secuencias antes de comparar
            method: Método DTW a usar
            
        Returns:
            Distancia DTW y camino de alineación
        """
        # Extraer secuencias
        seq1 = self.extract_pose_sequence(shot_id1)
        seq2 = self.extract_pose_sequence(shot_id2)
        
        if seq1 is None or seq2 is None or len(seq1) == 0 or len(seq2) == 0:
            print(f"No se pueden extraer secuencias para la comparación entre {shot_id1} y {shot_id2}")
            return float('inf'), []
        
        # Normalizar si es necesario
        if normalize:
            seq1 = self.normalize_sequence(seq1)
            seq2 = self.normalize_sequence(seq2)
        
        # Calcular DTW
        return self.calculate_dtw_distance(seq1, seq2, method=method)
    
    def compare_against_reference(self, input_shot_id, reference_type=None, normalize=True, method='fastdtw'):
        """
        Compara un golpe contra golpes de referencia del mismo tipo o especificados.
        
        Args:
            input_shot_id: ID del golpe a comparar
            reference_type: Tipo de referencias a usar (None=todas, 'same'=mismo tipo, 
                            o tupla (stroke_type, spin_type, player))
            normalize: Si se normalizan las secuencias antes de comparar
            method: Método DTW a usar
            
        Returns:
            Lista de tuplas (shot_id, distancia, metadata) ordenadas por similitud
        """
        if input_shot_id not in self.keypoints_data:
            print(f"El golpe {input_shot_id} no está cargado")
            return []
        
        input_meta = self.shot_metadata.get(input_shot_id, {})
        results = []
        
        # Determinar golpes de referencia a usar
        reference_shots = []
        
        if reference_type == 'same' and input_meta:
            # Usar solo golpes del mismo tipo y efecto
            stroke_type = input_meta.get('stroke_type')
            spin_type = input_meta.get('spin_type')
            
            for shot_id, meta in self.shot_metadata.items():
                if (meta.get('stroke_type') == stroke_type and 
                    meta.get('spin_type') == spin_type and 
                    shot_id != input_shot_id):
                    reference_shots.append(shot_id)
        elif isinstance(reference_type, tuple) and len(reference_type) == 3:
            # Usar golpes que coincidan con los criterios (stroke, spin, player)
            stroke, spin, player = reference_type
            
            for shot_id, meta in self.shot_metadata.items():
                if shot_id == input_shot_id:
                    continue
                    
                match = True
                if stroke and meta.get('stroke_type') != stroke:
                    match = False
                if spin and meta.get('spin_type') != spin:
                    match = False
                if player and player.lower() not in meta.get('player', '').lower():
                    match = False
                    
                if match:
                    reference_shots.append(shot_id)
        else:
            # Usar todos los golpes excepto el input
            reference_shots = [shot_id for shot_id in self.keypoints_data.keys() 
                              if shot_id != input_shot_id]
        
        # Comparar con cada referencia
        for ref_id in reference_shots:
            distance, _ = self.compare_shots(input_shot_id, ref_id, normalize, method)
            results.append((ref_id, distance, self.shot_metadata.get(ref_id, {})))
        
        # Ordenar por distancia (menor = más similar)
        results.sort(key=lambda x: x[1])
        return results
    
    def find_most_similar_shots(self, input_shot_id, top_n=3, reference_type=None):
        """
        Encuentra los golpes más similares a un golpe de referencia.
        
        Args:
            input_shot_id: ID del golpe de referencia
            top_n: Número de golpes similares a encontrar
            reference_type: Tipo de referencias a usar
            
        Returns:
            Lista de tuplas (shot_id, distancia, metadata) ordenadas por similitud
        """
        results = self.compare_against_reference(input_shot_id, reference_type)
        return results[:top_n]
    
    def analyze_player_style(self, player_name, stroke_type=None, spin_type=None):
        """
        Analiza el estilo de un jugador comparando sus golpes con los de otros jugadores.
        
        Args:
            player_name: Nombre del jugador a analizar
            stroke_type: Tipo de golpe a analizar (None = todos)
            spin_type: Tipo de efecto a analizar (None = todos)
            
        Returns:
            DataFrame con resultados del análisis
        """
        # Cargar golpes del jugador si no están ya cargados
        player_shots = self.load_shots_by_category(stroke_type, spin_type, player_name)
        
        if not player_shots:
            print(f"No se encontraron golpes para {player_name}")
            return None
        
        # Cargar golpes de otros jugadores para comparación
        other_players = set()
        for _, meta in self.shot_metadata.items():
            p = meta.get('player', '')
            if p and player_name.lower() not in p.lower():
                other_players.add(p)
        
        for other_player in other_players:
            self.load_shots_by_category(stroke_type, spin_type, other_player)
        
        # Realizar análisis
        analysis_data = []
        
        for shot_id in player_shots:
            shot_meta = self.shot_metadata.get(shot_id, {})
            stroke = shot_meta.get('stroke_type', 'Unknown')
            spin = shot_meta.get('spin_type', 'Unknown')
            
            # Encontrar golpes similares de otros jugadores
            similar_shots = self.find_most_similar_shots(
                shot_id, 
                top_n=5, 
                reference_type=(stroke, spin, None)  # Mismo tipo y efecto, cualquier jugador
            )
            
            for similar_id, distance, similar_meta in similar_shots:
                similar_player = similar_meta.get('player', 'Unknown')
                if player_name.lower() not in similar_player.lower():
                    analysis_data.append({
                        'player_shot': shot_id,
                        'stroke_type': stroke,
                        'spin_type': spin,
                        'similar_shot': similar_id,
                        'similar_player': similar_player,
                        'distance': distance
                    })
        
        if not analysis_data:
            print(f"No se pudieron encontrar comparaciones para {player_name}")
            return None
            
        # Crear DataFrame
        df = pd.DataFrame(analysis_data)
        
        # Calcular estadísticas agregadas
        summary = df.groupby(['stroke_type', 'spin_type', 'similar_player']).agg(
            avg_distance=('distance', 'mean'),
            min_distance=('distance', 'min'),
            count=('distance', 'count')
        ).reset_index()
        
        return summary
    
    def compare_all_shots(self, filter_criteria=None, normalize=True, method='fastdtw'):
        """
        Compara todos los golpes cargados entre sí usando DTW.
        
        Args:
            filter_criteria: Función que recibe metadata y devuelve True/False para incluir el golpe
            normalize: Si se normalizan las secuencias antes de comparar
            method: Método DTW a usar
            
        Returns:
            Matriz de distancias entre golpes y lista de IDs
        """
        # Filtrar golpes según criterios
        if filter_criteria is None:
            shot_ids = list(self.keypoints_data.keys())
        else:
            shot_ids = [
                shot_id for shot_id, meta in self.shot_metadata.items()
                if filter_criteria(meta)
            ]
        
        n_shots = len(shot_ids)
        if n_shots == 0:
            print("No hay golpes que cumplan con los criterios")
            return np.array([]), []
            
        distance_matrix = np.zeros((n_shots, n_shots))
        
        # Extraer y normalizar todas las secuencias primero
        sequences = {}
        for i, shot_id in enumerate(shot_ids):
            seq = self.extract_pose_sequence(shot_id)
            if normalize and len(seq) > 0:
                seq = self.normalize_sequence(seq)
            sequences[shot_id] = seq
        
        # Calcular todas las distancias
        for i, shot_id1 in enumerate(shot_ids):
            for j, shot_id2 in enumerate(shot_ids):
                if i <= j:  # Solo calculamos mitad de la matriz (es simétrica)
                    seq1 = sequences[shot_id1]
                    seq2 = sequences[shot_id2]
                    if len(seq1) == 0 or len(seq2) == 0:
                        distance = float('inf')
                    else:
                        distance, _ = self.calculate_dtw_distance(seq1, seq2, method=method)
                    distance_matrix[i, j] = distance
                    if i != j:  # Rellenar la otra mitad
                        distance_matrix[j, i] = distance
        
        return distance_matrix, shot_ids
    
    def visualize_distance_matrix(self, distance_matrix, shot_ids, title=None, figsize=(12, 10)):
        """
        Visualiza la matriz de distancias como un mapa de calor.
        
        Args:
            distance_matrix: Matriz de distancias DTW
            shot_ids: Lista de identificadores de golpes
            title: Título del gráfico
            figsize: Tamaño de la figura
        """
        if len(shot_ids) == 0 or distance_matrix.size == 0:
            print("No hay datos para visualizar")
            return
        
        # Crear etiquetas más legibles
        labels = []
        for shot_id in shot_ids:
            meta = self.shot_metadata.get(shot_id, {})
            player = meta.get('player', '').split()[0] if meta.get('player') else ''
            stroke = meta.get('stroke_type', '')
            spin = meta.get('spin_type', '')
            
            if player and stroke and spin:
                label = f"{player}-{stroke[:2]}-{spin[:2]}"
            else:
                label = shot_id.split('_')[-1]
            
            labels.append(label)
        
        plt.figure(figsize=figsize)
        
        # Filtrar valores infinitos para mejor visualización
        matrix_for_vis = distance_matrix.copy()
        if np.isinf(matrix_for_vis).any():
            max_finite = np.max(matrix_for_vis[~np.isinf(matrix_for_vis)])
            matrix_for_vis[np.isinf(matrix_for_vis)] = max_finite * 1.5
        
        # Normalizar para mejor visualización
        if np.max(matrix_for_vis) > 0:
            matrix_for_vis = matrix_for_vis / np.max(matrix_for_vis)
        
        sns.heatmap(matrix_for_vis, annot=False, fmt=".2f", 
                   xticklabels=labels, yticklabels=labels, cmap="viridis")
        
        if title:
            plt.title(title)
        else:
            plt.title('Matriz de Distancias DTW entre Golpes')
            
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def visualize_sequence(self, shot_id, frames=None):
        """
        Visualiza la secuencia de movimiento para un golpe específico.
        
        Args:
            shot_id: ID del golpe
            frames: Lista de índices de frames a visualizar, o None para todos
        """
        if shot_id not in self.keypoints_data:
            print(f"El golpe {shot_id} no está cargado")
            return
        
        sequence = self.extract_pose_sequence(shot_id)
        if len(sequence) == 0:
            print(f"No se pudieron extraer keypoints del golpe {shot_id}")
            return
        
        # Si no se especifican frames, seleccionamos algunos uniformemente
        if frames is None:
            n_frames = min(5, len(sequence))
            frames = np.linspace(0, len(sequence)-1, n_frames, dtype=int)
        
        fig, axes = plt.subplots(1, len(frames), figsize=(15, 5))
        if len(frames) == 1:
            axes = [axes]
        
        for i, frame_idx in enumerate(frames):
            if frame_idx >= len(sequence):
                continue
                
            # Obtener coordenadas de este frame
            coords = sequence[frame_idx].reshape(-1, 2)
            
            # Dibujar las conexiones entre keypoints
            edges = [
                # Cara
                (0, 1), (0, 2), (1, 3), (2, 4),
                # Torso
                (5, 6), (5, 11), (6, 12), (11, 12),
                # Brazos
                (5, 7), (7, 9), (6, 8), (8, 10),
                # Piernas
                (11, 13), (13, 15), (12, 14), (14, 16)
            ]
            
            axes[i].scatter(coords[:, 0], coords[:, 1], s=20)
            for edge in edges:
                if not (np.isnan(coords[edge[0]]).any() or np.isnan(coords[edge[1]]).any()):
                    axes[i].plot(coords[edge, 0], coords[edge, 1], 'b-')
            
            axes[i].set_title(f"Frame {frame_idx}")
            axes[i].invert_yaxis()  # Invertir eje Y para que coincida con coordenadas de imagen
            axes[i].axis('equal')
        
        # Obtener metadata para el título
        meta = self.shot_metadata.get(shot_id, {})
        player = meta.get('player', '')
        stroke = meta.get('stroke_type', '')
        spin = meta.get('spin_type', '')
        
        title = f"Secuencia de movimiento: {shot_id}"
        if player and stroke and spin:
            title = f"{player} - {stroke} {spin}"
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def visualize_alignment(self, shot_id1, shot_id2):
        """
        Visualiza la alineación DTW entre dos secuencias.
        
        Args:
            shot_id1, shot_id2: IDs de los golpes a comparar
        """
        seq1 = self.extract_pose_sequence(shot_id1)
        seq2 = self.extract_pose_sequence(shot_id2)
        
        if len(seq1) == 0 or len(seq2) == 0:
            print("No se pueden extraer secuencias para la comparación")
            return
        
        # Normalizar
        norm_seq1 = self.normalize_sequence(seq1)
        norm_seq2 = self.normalize_sequence(seq2)
        
        # Calcular DTW
        distance, path = self.calculate_dtw_distance(norm_seq1, norm_seq2)
        
        # Obtener metadata para los títulos
        meta1 = self.shot_metadata.get(shot_id1, {})
        meta2 = self.shot_metadata.get(shot_id2, {})
        
        player1 = meta1.get('player', '').split()[0] if meta1.get('player') else ''
        player2 = meta2.get('player', '').split()[0] if meta2.get('player') else ''
        
        stroke1 = meta1.get('stroke_type', '')
        stroke2 = meta2.get('stroke_type', '')
        
        spin1 = meta1.get('spin_type', '')
        spin2 = meta2.get('spin_type', '')
        
        label1 = f"{player1} {stroke1} {spin1}" if player1 and stroke1 and spin1 else shot_id1
        label2 = f"{player2} {stroke2} {spin2}" if player2 and stroke2 and spin2 else shot_id2
        
        # Usar PCA para reducir dimensiones para visualización
        pca = PCA(n_components=2)
        combined = np.vstack([norm_seq1, norm_seq2])
        pca_result = pca.fit_transform(combined)
        
        seq1_pca = pca_result[:len(norm_seq1)]
        seq2_pca = pca_result[len(norm_seq1):]
        
        # Visualizar
        plt.figure(figsize=(12, 10))
        
        # Dibujar secuencias en espacio PCA
        plt.subplot(2, 1, 1)
        plt.scatter(seq1_pca[:, 0], seq1_pca[:, 1], c=np.arange(len(seq1_pca)), cmap='Blues', label=label1)
        plt.scatter(seq2_pca[:, 0], seq2_pca[:, 1], c=np.arange(len(seq2_pca)), cmap='Reds', label=label2)
        plt.colorbar(label='Índice de frame')
        plt.title(f'Proyección PCA de secuencias (DTW distancia = {distance:.2f})')
        plt.legend()
        
        # Dibujar matriz de distancias y camino DTW
        if path:
            plt.subplot(2, 1, 2)
            # Calcular matriz de distancias
            dists = np.zeros((len(norm_seq1), len(norm_seq2)))
            for i in range(len(norm_seq1)):
                for j in range(len(norm_seq2)):
                    dists[i, j] = euclidean(norm_seq1[i], norm_seq2[j])
            
            plt.imshow(dists, cmap='viridis', aspect='auto')
            plt.colorbar(label='Distancia euclidiana')
            
            # Dibujar camino DTW
            path = np.array(path)
            plt.plot(path[:, 1], path[:, 0], 'r-')
            plt.title('Matriz de distancias y camino DTW')
            plt.xlabel(f'Secuencia {label2}')
            plt.ylabel(f'Secuencia {label1}')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_keyframes(self, shot_id):
        """
        Visualiza los keyframes importantes de un golpe (preparación, impacto, etc).
        
        Args:
            shot_id: ID del golpe
        """
        if shot_id not in self.keypoints_data:
            print(f"El golpe {shot_id} no está cargado")
            return
        
        data = self.keypoints_data[shot_id]
        keyframes = []
        
        # Buscar frames con etiqueta de keyframe_type
        for i, frame_data in enumerate(data):
            if isinstance(frame_data, dict) and 'keyframe_type' in frame_data:
                keyframes.append((i, frame_data['keyframe_type'], frame_data))
        
        if not keyframes:
            print(f"No se encontraron keyframes etiquetados para {shot_id}")
            return
        
        # Visualizar los keyframes
        fig, axes = plt.subplots(1, len(keyframes), figsize=(15, 5))
        if len(keyframes) == 1:
            axes = [axes]
        
        for i, (_, keyframe_type, frame_data) in enumerate(keyframes):
            # Obtener keypoints
            keypoints = frame_data.get('keypoints', [])
            if not keypoints:
                continue
                
            # Extraer coordenadas
            coords = np.array([[kp.get('x', 0), kp.get('y', 0)] for kp in keypoints])
            
            # Dibujar las conexiones entre keypoints
            edges = [
                # Cara
                (0, 1), (0, 2), (1, 3), (2, 4),
                # Torso
                (5, 6), (5, 11), (6, 12), (11, 12),
                # Brazos
                (5, 7), (7, 9), (6, 8), (8, 10),
                # Piernas
                (11, 13), (13, 15), (12, 14), (14, 16)
            ]
            
            axes[i].scatter(coords[:, 0], coords[:, 1], s=20)
            for edge in edges:
                if (edge[0] < len(coords) and edge[1] < len(coords) and 
                    not (np.isnan(coords[edge[0]]).any() or np.isnan(coords[edge[1]]).any())):
                    axes[i].plot(coords[edge, 0], coords[edge, 1], 'b-')
            
            axes[i].set_title(f"{keyframe_type}")
            axes[i].invert_yaxis()  # Invertir eje Y para que coincida con coordenadas de imagen
            axes[i].axis('equal')
        
        # Obtener metadata para el título
        meta = self.shot_metadata.get(shot_id, {})
        player = meta.get('player', '')
        stroke = meta.get('stroke_type', '')
        spin = meta.get('spin_type', '')
        
        title = f"Keyframes: {shot_id}"
        if player and stroke and spin:
            title = f"Keyframes: {player} - {stroke} {spin}"
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def analyze_stroke_phases(self, shot_id, num_phases=5):
        """
        Analiza un golpe dividiéndolo en fases y visualizando cada fase.
        
        Args:
            shot_id: ID del golpe
            num_phases: Número de fases a dividir el golpe
        """
        sequence = self.extract_pose_sequence(shot_id)
        if len(sequence) == 0:
            print(f"No se pudieron extraer keypoints del golpe {shot_id}")
            return
        
        # Dividir la secuencia en fases
        phase_indices = np.linspace(0, len(sequence)-1, num_phases, dtype=int)
        
        # Visualizar las fases
        self.visualize_sequence(shot_id, frames=phase_indices)
        
        # Análisis adicional: calcular velocidad de movimiento entre fases
        velocities = []
        for i in range(len(phase_indices)-1):
            frame1 = sequence[phase_indices[i]]
            frame2 = sequence[phase_indices[i+1]]
            
            # Calcular distancia euclidiana media entre todos los keypoints
            frame1_reshaped = frame1.reshape(-1, 2)
            frame2_reshaped = frame2.reshape(-1, 2)
            
            valid_points = (~np.isnan(frame1_reshaped).any(axis=1)) & (~np.isnan(frame2_reshaped).any(axis=1))
            if sum(valid_points) > 0:
                dists = np.linalg.norm(frame2_reshaped[valid_points] - frame1_reshaped[valid_points], axis=1)
                avg_dist = np.mean(dists)
                velocities.append(avg_dist)
        
        # Mostrar velocidades
        if velocities:
            plt.figure(figsize=(10, 4))
            plt.bar(range(1, len(velocities)+1), velocities)
            plt.xlabel('Fase')
            plt.ylabel('Velocidad promedio')
            plt.title(f'Velocidad de movimiento por fase: {shot_id}')
            plt.tight_layout()
            plt.show()
    
    def compare_by_categories(self, categories=None):
        """
        Compara golpes agrupándolos por categorías (tipo de golpe, efecto, jugador).
        
        Args:
            categories: Lista de categorías para agrupar ('stroke_type', 'spin_type', 'player')
                      Si es None, usa ['stroke_type', 'spin_type']
        
        Returns:
            DataFrame con resultados de comparación entre categorías
        """
        if categories is None:
            categories = ['stroke_type', 'spin_type']
        
        # Verificar que las categorías son válidas
        valid_categories = ['stroke_type', 'spin_type', 'player']
        categories = [cat for cat in categories if cat in valid_categories]
        
        if not categories:
            print("No se especificaron categorías válidas")
            return None
        
        # Agrupar golpes por categorías
        category_groups = {}
        
        for shot_id, meta in self.shot_metadata.items():
            # Crear clave de categoría
            key_parts = []
            for cat in categories:
                key_parts.append(meta.get(cat, 'Unknown'))
            
            key = ' - '.join(key_parts)
            
            if key not in category_groups:
                category_groups[key] = []
            
            category_groups[key].append(shot_id)
        
        # Comparar entre grupos
        results = []
        
        for key1, shots1 in category_groups.items():
            for key2, shots2 in category_groups.items():
                if key1 >= key2:  # Evitar duplicados (y comparar consigo mismo)
                    continue
                    
                # Calcular distancia DTW promedio entre todos los pares
                distances = []
                
                for shot_id1 in shots1:
                    for shot_id2 in shots2:
                        dist, _ = self.compare_shots(shot_id1, shot_id2)
                        if not np.isinf(dist):
                            distances.append(dist)
                
                if distances:
                    avg_dist = np.mean(distances)
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    
                    results.append({
                        'category1': key1,
                        'category2': key2,
                        'avg_distance': avg_dist,
                        'min_distance': min_dist,
                        'max_distance': max_dist,
                        'num_comparisons': len(distances)
                    })
        
        if not results:
            print("No se pudieron realizar comparaciones entre categorías")
            return None
            
        return pd.DataFrame(results)
    
    def cluster_shots(self, n_clusters=3, method='kmeans'):
        """
        Agrupa golpes similares usando clustering.
        
        Args:
            n_clusters: Número de clusters a crear
            method: Método de clustering ('kmeans', 'hierarchical', 'dbscan')
            
        Returns:
            DataFrame con resultados de clustering
        """
        from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
        from sklearn import metrics
        
        # Obtener matriz de distancias
        distance_matrix, shot_ids = self.compare_all_shots()
        
        if len(shot_ids) == 0:
            print("No hay suficientes datos para clustering")
            return None
        
        # Reemplazar valores infinitos con un valor grande
        if np.isinf(distance_matrix).any():
            max_finite = np.max(distance_matrix[~np.isinf(distance_matrix)])
            distance_matrix[np.isinf(distance_matrix)] = max_finite * 2
        
        # Para métodos basados en distancia, usamos MDS para crear un espacio euclidiano
        from sklearn.manifold import MDS
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        pos = mds.fit_transform(distance_matrix)
        
        # Aplicar clustering
        labels = None
        
        if method == 'kmeans':
            clustering = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clustering.fit_predict(pos)
        elif method == 'hierarchical':
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clustering.fit_predict(pos)
        elif method == 'dbscan':
            # DBSCAN determina automáticamente el número de clusters
            clustering = DBSCAN(eps=0.5, min_samples=2)
            labels = clustering.fit_predict(pos)
        
        if labels is None:
            print(f"Método de clustering no reconocido: {method}")
            return None
        
        # Crear DataFrame con resultados
        results = []
        
        for i, shot_id in enumerate(shot_ids):
            meta = self.shot_metadata.get(shot_id, {})
            
            results.append({
                'shot_id': shot_id,
                'cluster': int(labels[i]),
                'player': meta.get('player', 'Unknown'),
                'stroke_type': meta.get('stroke_type', 'Unknown'),
                'spin_type': meta.get('spin_type', 'Unknown'),
                'x': pos[i, 0],
                'y': pos[i, 1]
            })
        
        df = pd.DataFrame(results)
        
        # Visualizar resultados
        plt.figure(figsize=(12, 8))
        
        # Colores para clusters
        colors = plt.cm.rainbow(np.linspace(0, 1, len(df['cluster'].unique())))
        
        # Marcadores para tipos de golpe
        markers = {'Backhand': 'o', 'Forehand': 's', 'Serve': '^', 'Smash': 'D', 'Unknown': 'x'}
        
        # Agrupar por cluster y tipo de golpe
        for i, (cluster, group) in enumerate(df.groupby('cluster')):
            for stroke, subgroup in group.groupby('stroke_type'):
                marker = markers.get(stroke, 'o')
                plt.scatter(
                    subgroup['x'], subgroup['y'], 
                    color=colors[i], 
                    marker=marker,
                    s=100,
                    label=f"Cluster {cluster}, {stroke}"
                )
        
        # Añadir etiquetas para cada punto
        for _, row in df.iterrows():
            player = row['player'].split()[0] if row['player'] else ''
            stroke = row['stroke_type'][:2] if row['stroke_type'] else ''
            spin = row['spin_type'][:2] if row['spin_type'] else ''
            
            label = f"{player}-{stroke}-{spin}"
            plt.annotate(label, (row['x'], row['y']), fontsize=8)
        
        plt.title(f"Clustering de golpes usando {method}")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        plt.tight_layout()
        plt.show()
        
        return df
    
    def extract_keypoints_statistics(self, shot_id):
        """
        Extrae estadísticas de los keypoints para un golpe específico.
        
        Args:
            shot_id: ID del golpe
            
        Returns:
            DataFrame con estadísticas de keypoints
        """
        if shot_id not in self.keypoints_data:
            print(f"El golpe {shot_id} no está cargado")
            return None
        
        sequence = self.extract_pose_sequence(shot_id)
        if len(sequence) == 0:
            print(f"No se pudieron extraer keypoints del golpe {shot_id}")
            return None
        
        # Reshape para tener pares (x,y)
        reshaped_sequence = sequence.reshape(sequence.shape[0], -1, 2)
        
        # Calcular estadísticas para cada keypoint
        stats = []
        
        for i, keypoint_name in enumerate(self.keypoint_names):
            if i >= reshaped_sequence.shape[1]:
                continue
                
            # Extraer coordenadas x, y para este keypoint a lo largo de la secuencia
            x_coords = reshaped_sequence[:, i, 0]
            y_coords = reshaped_sequence[:, i, 1]
            
            # Filtrar valores NaN
            valid_x = x_coords[~np.isnan(x_coords)]
            valid_y = y_coords[~np.isnan(y_coords)]
            
            if len(valid_x) == 0 or len(valid_y) == 0:
                continue
            
            # Calcular estadísticas
            x_mean = np.mean(valid_x)
            y_mean = np.mean(valid_y)
            x_std = np.std(valid_x)
            y_std = np.std(valid_y)
            
            # Calcular distancia recorrida
            x_diff = np.diff(valid_x)
            y_diff = np.diff(valid_y)
            dist = np.sum(np.sqrt(x_diff**2 + y_diff**2))
            
            # Velocidad promedio (distancia / número de frames)
            velocity = dist / (len(valid_x) - 1) if len(valid_x) > 1 else 0
            
            # Calcular aceleración promedio
            if len(valid_x) > 2:
                x_accel = np.diff(x_diff)
                y_accel = np.diff(y_diff)
                accel = np.sum(np.sqrt(x_accel**2 + y_accel**2)) / (len(valid_x) - 2)
            else:
                accel = 0
            
            stats.append({
                'keypoint': keypoint_name,
                'x_mean': x_mean,
                'y_mean': y_mean,
                'x_std': x_std,
                'y_std': y_std,
                'distance': dist,
                'velocity': velocity,
                'acceleration': accel
            })
        
        return pd.DataFrame(stats)

    def analyze_shot_differences(self, shot_id1, shot_id2):
        """
        Analiza las diferencias entre dos golpes.
        
        Args:
            shot_id1, shot_id2: IDs de los golpes a comparar
            
        Returns:
            DataFrame con diferencias entre keypoints
        """
        # Extraer estadísticas de cada golpe
        stats1 = self.extract_keypoints_statistics(shot_id1)
        stats2 = self.extract_keypoints_statistics(shot_id2)
        
        if stats1 is None or stats2 is None:
            return None
        
        # Combinar y calcular diferencias
        merged = pd.merge(stats1, stats2, on='keypoint', suffixes=('_1', '_2'))
        
        # Calcular diferencias absolutas y porcentuales
        for col in ['x_mean', 'y_mean', 'x_std', 'y_std', 'distance', 'velocity', 'acceleration']:
            merged[f'{col}_diff'] = merged[f'{col}_1'] - merged[f'{col}_2']
            merged[f'{col}_pct_diff'] = (
                100 * (merged[f'{col}_1'] - merged[f'{col}_2']) / 
                merged[f'{col}_2'].replace(0, 1e-10)
            )
        
        # Visualizar las diferencias más significativas
        plt.figure(figsize=(12, 8))
        
        # Enfocarse en las métricas de velocidad y aceleración
        subset = merged[['keypoint', 'velocity_diff', 'acceleration_diff']]
        
        # Crear un heatmap
        pivot = subset.pivot(index='keypoint', values=['velocity_diff', 'acceleration_diff'])
        sns.heatmap(pivot, annot=True, cmap='coolwarm', center=0)
        
        meta1 = self.shot_metadata.get(shot_id1, {})
        meta2 = self.shot_metadata.get(shot_id2, {})
        
        plt.title(f"Diferencias entre {shot_id1} y {shot_id2}")
        plt.tight_layout()
        plt.show()
        
        return merged

    def create_reference_shots_report(self, output_file=None):
        """
        Genera un informe de comparación entre todos los golpes de referencia.
        
        Args:
            output_file: Nombre del archivo CSV para guardar el informe
            
        Returns:
            DataFrame con resultados de comparación
        """
        if not self.reference_data:
            print("No hay datos de referencia cargados. Primero ejecuta load_reference_file().")
            return None
        
        # Asegurarse de que todos los golpes de referencia estén cargados
        self.load_all_reference_shots()
        
        # Extraer todos los IDs de golpes de referencia
        reference_shot_ids = []
        for stroke_type, spins in self.reference_data.items():
            for spin_type, players in spins.items():
                for _, shot_id in players.items():
                    if shot_id in self.keypoints_data:
                        reference_shot_ids.append(shot_id)
        
        # Crear matriz de comparación
        n_shots = len(reference_shot_ids)
        comparison_data = []
        
        for i, shot_id1 in enumerate(reference_shot_ids):
            meta1 = self.shot_metadata.get(shot_id1, {})
            
            for j, shot_id2 in enumerate(reference_shot_ids):
                if i >= j:  # Solo la mitad superior de la matriz
                    continue
                    
                meta2 = self.shot_metadata.get(shot_id2, {})
                
                # Calcular distancia DTW
                distance, _ = self.compare_shots(shot_id1, shot_id2)
                
                # Guardar resultados
                comparison_data.append({
                    'shot_id1': shot_id1,
                    'player1': meta1.get('player', 'Unknown'),
                    'stroke_type1': meta1.get('stroke_type', 'Unknown'),
                    'spin_type1': meta1.get('spin_type', 'Unknown'),
                    'shot_id2': shot_id2,
                    'player2': meta2.get('player', 'Unknown'),
                    'stroke_type2': meta2.get('stroke_type', 'Unknown'),
                    'spin_type2': meta2.get('spin_type', 'Unknown'),
                    'dtw_distance': distance
                })
        
        if not comparison_data:
            print("No se pudieron realizar comparaciones entre golpes de referencia")
            return None
            
        # Crear DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Guardar a CSV si se especificó un archivo
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Informe guardado en {output_file}")
        
        return df
    
    def benchmark_against_reference(self, input_shot_id):
        """
        Compara un golpe contra todos los tipos de referencia adecuados.
        
        Args:
            input_shot_id: ID del golpe a comparar
            
        Returns:
            DataFrame con resultados de benchmark
        """
        if input_shot_id not in self.keypoints_data:
            print(f"El golpe {input_shot_id} no está cargado")
            return None
        
        input_meta = self.shot_metadata.get(input_shot_id, {})
        
        if not input_meta:
            print(f"No hay metadata disponible para {input_shot_id}")
            return None
        
        # Encontrar referencias del mismo tipo de golpe y efecto
        stroke_type = input_meta.get('stroke_type')
        spin_type = input_meta.get('spin_type')
        
        if not stroke_type or not spin_type:
            print(f"Tipo de golpe o efecto no disponible para {input_shot_id}")
            return None
        
        # Cargar golpes de referencia si aún no están cargados
        if not self.reference_data:
            self.load_reference_file()
            self.load_all_reference_shots()
        
        # Buscar referencias adecuadas
        reference_shots = []
        
        for s_type, spins in self.reference_data.items():
            if stroke_type not in s_type:
                continue
                
            if spin_type in spins:
                for player, shot_id in spins[spin_type].items():
                    if shot_id != input_shot_id and shot_id in self.keypoints_data:
                        reference_shots.append((shot_id, player))
        
        if not reference_shots:
            print(f"No se encontraron golpes de referencia para {stroke_type} {spin_type}")
            return None
        
        # Comparar con cada referencia
        results = []
        
        for ref_id, player in reference_shots:
            distance, _ = self.compare_shots(input_shot_id, ref_id)
            
            results.append({
                'reference_shot': ref_id,
                'reference_player': player,
                'dtw_distance': distance
            })
        
        # Ordenar por distancia
        results_df = pd.DataFrame(results).sort_values(by='dtw_distance')
        
        # Visualizar resultados
        plt.figure(figsize=(10, 6))
        sns.barplot(x='reference_player', y='dtw_distance', data=results_df)
        plt.title(f"Comparación de {input_shot_id} con golpes de referencia")
        plt.xlabel('Jugador de referencia')
        plt.ylabel('Distancia DTW (menor = más similar)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return results_df

 
if __name__ == "__main__":
    # Crear el comparador
    base_dir = "/Users/emilio/Documents/Winner Way/data/json"  # Cambiar esto a tu directorio
    comparator = TennisMotionComparator(base_dir)
    
    # Cargar archivo de referencia
    comparator.load_reference_file("reference_all.json")
    
    # Cargar golpes de un tipo específico
    backhand_shots = comparator.load_shots_by_category(stroke_type="Backhand", spin_type="Drop Shot")
    
    # Cargar todos los golpes de referencia
    comparator.load_all_reference_shots()
    
    # Ejemplo: Comparar dos golpes
    if len(backhand_shots) >= 2:
        shot1 = backhand_shots[0]
        shot2 = backhand_shots[1]
        
        print(f"Comparando {shot1} y {shot2}:")
        distance, _ = comparator.compare_shots(shot1, shot2)
        print(f"  Distancia DTW: {distance:.2f}")
        
        # Visualizar la alineación
        comparator.visualize_alignment(shot1, shot2)
        
        # Analizar diferencias
        comparator.analyze_shot_differences(shot1, shot2)
    
    # Ejemplo: Buscar golpes similares a uno de referencia
    if backhand_shots:
        reference_shot = backhand_shots[0]
        similar_shots = comparator.find_most_similar_shots(reference_shot, top_n=3)
        
        print(f"Golpes más similares a {reference_shot}:")
        for shot_id, distance, meta in similar_shots:
            player = meta.get('player', 'Unknown')
            print(f"  {shot_id} ({player}): {distance:.2f}")
    
    # Ejemplo: Comparar por categorías
    category_comparison = comparator.compare_by_categories(['stroke_type', 'spin_type'])
    if category_comparison is not None:
        print("\nComparación por categorías:")
        print(category_comparison.sort_values(by='avg_distance').head())
    
    # Ejemplo: Clustering de golpes
    comparator.cluster_shots(n_clusters=4)
    
    # Ejemplo: Informe de golpes de referencia