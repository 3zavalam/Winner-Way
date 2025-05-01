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
from typing import List, Optional, Union, Dict, Tuple, Any
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

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
        self.shot_metadata = {}
        self._distance_cache = {}  # Cache para distancias DTW calculadas
        
        self.keypoint_names = [
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
        path = self.data_dir / fname

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
        # Log para depuración
        logging.info(f"Buscando archivo: {pattern}")
        
        for p in self.json_dir.rglob(pattern):
            logging.info(f"Encontrado: {p}")
            return p
        
        # Si no se encontró, imprimir para depuración
        logging.error(f"No se encontró el archivo: {pattern}")
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
            
            # Determinar el stroke_type desde el shot_id
            stroke_type = None
            if "bh" in shot_id.lower():
                # Verificar si es Backhand 1H o 2H
                if "1" in shot_id.lower() or not any(x in shot_id.lower() for x in ["2", "two", "2h"]):
                    stroke_type = "Backhand 1H"
                else:
                    stroke_type = "Backhand 2H"
            elif "fh" in shot_id.lower():
                stroke_type = "Forehand"
            elif "sv" in shot_id.lower():
                stroke_type = "Serve"
            elif "sh" in shot_id.lower():
                stroke_type = "Smash"
            
            # Determinar el spin_type desde el shot_id
            spin_type = None
            if "ts" in shot_id.lower():
                spin_type = "Topspin"
            elif "ft" in shot_id.lower():
                spin_type = "Flat"
            elif "sl" in shot_id.lower():
                spin_type = "Slice"
            elif "ds" in shot_id.lower():
                spin_type = "Drop Shot"
            elif "kk" in shot_id.lower():
                spin_type = "Kick"
            
            # Si hay "rt" antes del tipo, es un return
            if "rt" in shot_id.lower() and spin_type:
                spin_type = f"Return {spin_type}"
            
            # Extraer player a partir del shot_id
            tokens = shot_id.split("_")
            player = f"{tokens[0].capitalize()} {tokens[1].capitalize()}" if len(tokens) >= 2 else None
            
            # Guardar metadata, usando información del shot_id en lugar de la estructura de carpetas
            self.shot_metadata[shot_id] = {
                "stroke_type": stroke_type,
                "spin_type": spin_type,
                "player": player,
                "file_path": str(file_path),
                "folder": str(file_path.parent)
            }
            
            logging.info(f"✓ Keypoints cargados: {shot_id} (Tipo: {stroke_type}, Efecto: {spin_type})")
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
            for spin_type, shot_ids in spins.items():
                # Si 'shot_ids' es una lista de identificadores de golpes, itera sobre ella
                for shot_id in shot_ids:
                    if self.load_keypoints_file(shot_id):
                        count += 1

        logging.info(f"✔ Total de keypoints de referencia cargados: {count}")
        return count

    def load_shots_by_category(self, stroke_type=None, spin_type=None, player=None) -> List[str]:
        """
        Carga todos los golpes de una categoría específica (por ejemplo, Backhand, Forehand)
        y también distingue entre 1H y 2H si corresponde usando la estructura de carpetas.
        Solo carga los clips dentro de la misma carpeta de tipo y spin.
        
        Args:
            stroke_type: Tipo de golpe a cargar (e.g., "Backhand", "Forehand")
            spin_type: Tipo de efecto a cargar (e.g., "Topspin", "Slice")
            player: Nombre del jugador (opcional)
            
        Returns:
            Lista de shot_ids cargados
        """
        shots = []
        
        # Construir la ruta base para buscar
        base_path = self.json_dir
        if stroke_type:
            base_path = base_path / stroke_type
        
        # Patrón para buscar archivos
        pattern = "*_keypoints.json"
        
        # Buscar en la estructura de directorios
        for p in base_path.rglob(pattern):
            # Verificar el tipo de efecto si está especificado
            if spin_type:
                # Si el spin_type está en la ruta, continuar; si no, saltarlo
                if spin_type not in p.parts:
                    continue
            
            # Extraer shot_id del nombre del archivo
            shot_id = p.stem.replace('_keypoints', '')
            
            # Filtrar por jugador si está especificado
            if player:
                # Extraer jugador del shot_id
                parts = shot_id.split('_')
                if len(parts) >= 2:
                    shot_player = f"{parts[0]} {parts[1]}".lower()
                    if player.lower() not in shot_player:
                        continue
            
            # Cargar el archivo y añadir a la lista si se cargó correctamente
            if self.load_keypoints_file(shot_id):
                shots.append(shot_id)
        
        logging.info(f"Se cargaron {len(shots)} golpes para la categoría: {stroke_type}/{spin_type}" +
                    (f", jugador: {player}" if player else ""))
        return shots

    def load_shots_from_folder(self, folder_path: Union[str, Path]) -> List[str]:
        """
        Carga todos los golpes dentro de una carpeta específica.
        
        Args:
            folder_path: Ruta a la carpeta que contiene los archivos _keypoints.json
            
        Returns:
            Lista de shot_ids cargados
        """
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            logging.error(f"La carpeta no existe: {folder_path}")
            return []
        
        loaded_shots = []
        
        # Buscar todos los archivos _keypoints.json en esta carpeta
        for file_path in folder_path.glob("*_keypoints.json"):
            shot_id = file_path.stem.replace('_keypoints', '')
            
            # Cargar el archivo
            if self.load_keypoints_file(shot_id):
                loaded_shots.append(shot_id)
        
        logging.info(f"Se cargaron {len(loaded_shots)} golpes desde {folder_path}")
        return loaded_shots

    def load_all_available_shots(self) -> int:
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
        
        logging.info(f"Se cargaron {loaded_count} archivos de keypoints disponibles.")
        return loaded_count
    
    def extract_pose_sequence(self, shot_id: str) -> np.ndarray:
        """
        Extrae la secuencia de poses de un golpe como una matriz numpy.
        
        Args:
            shot_id: Identificador del golpe
            
        Returns:
            Matriz numpy de forma (n_frames, n_keypoints*2) con coordenadas x,y concatenadas
        """
        if shot_id not in self.keypoints_data:
            logging.error(f"El golpe {shot_id} no está cargado")
            return np.array([])
        
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
    
    def normalize_sequence(self, sequence: np.ndarray) -> np.ndarray:
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

    def calculate_dtw_distance(self, seq1: np.ndarray, seq2: np.ndarray, method='fastdtw') -> Tuple[float, list]:
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
        
    def extract_shot_type(self, shot_id: str) -> Tuple[str, str]:
        """
        Extrae el tipo de golpe (stroke_type) y la variante del golpe (shot_variant) desde el shot_id.
        
        Args:
            shot_id: Identificador del golpe, como 'roger_federer_bh_ts1_01'.
        
        Returns:
            tuple: (stroke_type, shot_variant)
        """
        # Definir el tipo de golpe (stroke_type)
        if "sv" in shot_id.lower():
            stroke_type = "Serve"
        elif "fh" in shot_id.lower():
            stroke_type = "Forehand"
        elif "bh" in shot_id.lower():
            stroke_type = "Backhand"
        elif "sh" in shot_id.lower():
            stroke_type = "Smash"
        elif "lb" in shot_id.lower():
            stroke_type = "Lob"
        elif "vy" in shot_id.lower():
            stroke_type = "Volley"
        else:
            stroke_type = "Unknown"
        
        # Definir la variante del golpe (shot_variant)
        if "ts" in shot_id.lower():
            shot_variant = "Topspin"
        elif "ft" in shot_id.lower():
            shot_variant = "Flat"
        elif "sl" in shot_id.lower():
            shot_variant = "Slice"
        elif "kk" in shot_id.lower():
            shot_variant = "Kick"
        elif "ds" in shot_id.lower():
            shot_variant = "Drop Shot"
        elif "rt" in shot_id.lower():
            shot_variant = "Return"
        elif "oh" in shot_id.lower():
            shot_variant = "Overhead"
        elif "sw" in shot_id.lower():
            shot_variant = "Swing"
        elif "half" in shot_id.lower():
            shot_variant = "Half Volley"
        else:
            shot_variant = "Unknown"
        
        return stroke_type, shot_variant

    def compare_shots_by_category(self, shot_id1: str, shot_id2: str, normalize=True, method='fastdtw') -> Tuple[float, list]:
        """
        Compara dos golpes específicos usando DTW, asegurando que pertenezcan a la misma categoría.
        
        Args:
            shot_id1, shot_id2: Identificadores de los golpes a comparar
            normalize: Si se normalizan las secuencias antes de comparar
            method: Método DTW a usar
            
        Returns:
            Tuple con la distancia DTW y camino de alineación, o (float('inf'), []) si no son compatibles
        """
        # Verificar que ambos golpes estén cargados
        if shot_id1 not in self.keypoints_data or shot_id2 not in self.keypoints_data:
            logging.error(f"Uno o ambos golpes no están cargados: {shot_id1}, {shot_id2}")
            return float('inf'), []
        
        # Extraer metadata
        meta1 = self.shot_metadata.get(shot_id1, {})
        meta2 = self.shot_metadata.get(shot_id2, {})
        
        # Verificar que sean de la misma categoría
        if (meta1.get('stroke_type') != meta2.get('stroke_type') or 
            meta1.get('spin_type') != meta2.get('spin_type')):
            logging.info(f"Los golpes son de diferentes categorías: {shot_id1} ({meta1.get('stroke_type')}, {meta1.get('spin_type')}) vs. {shot_id2} ({meta2.get('stroke_type')}, {meta2.get('spin_type')})")
            return float('inf'), []
        
        # Verificar que estén en la misma ruta de carpeta (mismo nivel de estructura)
        folder1 = meta1.get('folder', '')
        folder2 = meta2.get('folder', '')
        
        if folder1 != folder2:
            logging.info(f"Los golpes están en diferentes carpetas: {folder1} vs. {folder2}")
            return float('inf'), []
        
        # Si pasaron todas las verificaciones, proceder con la comparación
        # Extraer secuencias
        seq1 = self.extract_pose_sequence(shot_id1)
        seq2 = self.extract_pose_sequence(shot_id2)
        
        if len(seq1) == 0 or len(seq2) == 0:
            logging.error(f"No se pueden extraer secuencias para la comparación entre {shot_id1} y {shot_id2}")
            return float('inf'), []
        
        # Normalizar si es necesario
        if normalize:
            seq1 = self.normalize_sequence(seq1)
            seq2 = self.normalize_sequence(seq2)
        
        # Calcular DTW
        return self.calculate_dtw_distance(seq1, seq2, method=method)

    def compare_shots(self, shot_id1: str, shot_id2: str, normalize=True, method='fastdtw') -> Tuple[float, list]:
        """
        Compara dos golpes específicos usando DTW. Mantiene compatibilidad con código anterior.
        
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
            logging.error(f"No se pueden extraer secuencias para la comparación entre {shot_id1} y {shot_id2}")
            return float('inf'), []

        # Verificación de los tipos de golpe
        stroke_type1, spin_type1 = self.extract_shot_type(shot_id1) 
        stroke_type2, spin_type2 = self.extract_shot_type(shot_id2)

        """if stroke_type1 != stroke_type2 or spin_type1 != spin_type2:
            logging.info(f"Advertencia: Los golpes no coinciden. {shot_id1} ({stroke_type1}, {spin_type1}) vs. {shot_id2} ({stroke_type2}, {spin_type2})")
        else:
            logging.info(f"Los golpes coinciden. {shot_id1} ({stroke_type1}, {spin_type1}) vs. {shot_id2} ({stroke_type2}, {spin_type2})")"""
        
        # Normalizar si es necesario
        if normalize:
            seq1 = self.normalize_sequence(seq1)
            seq2 = self.normalize_sequence(seq2)
        
        # Calcular DTW
        return self.calculate_dtw_distance(seq1, seq2, method=method)

    def get_or_calculate_distance(self, shot_id1: str, shot_id2: str, normalize=True, method='fastdtw') -> float:
        """
        Obtiene la distancia DTW entre dos golpes del caché o la calcula si no existe.
        
        Args:
            shot_id1, shot_id2: Identificadores de los golpes
            normalize: Si se normalizan las secuencias
            method: Método DTW
            
        Returns:
            Distancia DTW
        """
        # Ordenar IDs para consistencia en el caché
        if shot_id1 > shot_id2:
            shot_id1, shot_id2 = shot_id2, shot_id1
        
        # Clave de caché
        cache_key = f"{shot_id1}_{shot_id2}_{normalize}_{method}"
        
        # Verificar si ya existe en caché
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]
        
        # Calcular distancia
        distance, _ = self.compare_shots_by_category(shot_id1, shot_id2, normalize, method)
        
        # Guardar en caché
        self._distance_cache[cache_key] = distance
        
        return distance

    def compare_shots_parallel(self, shots: List[str], max_workers=4) -> List[Tuple[str, str, float]]:
        """
        Compara pares de golpes en paralelo.
        
        Args:
            shots: Lista de shot_ids a comparar
            max_workers: Número máximo de procesos
            
        Returns:
            Lista de resultados (shot_id1, shot_id2, distance)
        """
        pairs = list(itertools.combinations(shots, 2))
        results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_pair = {
                executor.submit(self.get_or_calculate_distance, pair[0], pair[1]): pair 
                for pair in pairs
            }
            
            for future in as_completed(future_to_pair):
                pair = future_to_pair[future]
                try:
                    distance = future.result()
                    if not np.isinf(distance):
                        results.append((pair[0], pair[1], distance))
                except Exception as e:
                    logging.error(f"Error al comparar {pair}: {e}")
        
        return results
    
    def compare_all_against_references(self, output_file=None):
        """
        Compara todos los golpes disponibles contra los golpes de referencia en reference_all.json.
        
        Args:
            output_file: Ruta para guardar el informe CSV
            
        Returns:
            DataFrame con los resultados de comparación
        """
        if not self.reference_data:
            logging.error("No hay datos de referencia cargados. Primero ejecuta load_reference_file().")
            return None
        
        # Cargar todos los golpes de referencia
        self.load_all_reference_shots()
        
        # Agrupar referencias por tipo de golpe y efecto
        reference_shots_by_category = {}
        for stroke_type, spins in self.reference_data.items():
            for spin_type, shot_ids in spins.items():
                key = (stroke_type, spin_type)
                if key not in reference_shots_by_category:
                    reference_shots_by_category[key] = []
                reference_shots_by_category[key].extend(shot_ids)
        
        # Log para depuración
        logging.info("Referencias por categoría:")
        for (stroke, spin), refs in reference_shots_by_category.items():
            logging.info(f"  {stroke} / {spin}: {len(refs)} golpes")
        
        # Cargar todos los golpes disponibles
        self.load_all_available_shots()
        
        # Agrupar todos los golpes por categoría
        all_shots_by_category = {}
        for shot_id, meta in self.shot_metadata.items():
            stroke_type = meta.get('stroke_type')
            spin_type = meta.get('spin_type')
            
            if stroke_type and spin_type:
                key = (stroke_type, spin_type)
                if key not in all_shots_by_category:
                    all_shots_by_category[key] = []
                all_shots_by_category[key].append(shot_id)
        
        # Log para depuración
        logging.info("Todos los golpes por categoría:")
        for (stroke, spin), shots in all_shots_by_category.items():
            logging.info(f"  {stroke} / {spin}: {len(shots)} golpes")
        
        # Comparar cada golpe con las referencias de su misma categoría
        comparisons = []
        
        for key, shots in all_shots_by_category.items():
            stroke_type, spin_type = key
            
            # Verificar si tenemos referencias para esta categoría
            if key in reference_shots_by_category:
                refs = reference_shots_by_category[key]
                logging.info(f"Comparando {len(shots)} golpes con {len(refs)} referencias en {stroke_type} {spin_type}")
                
                # Para cada golpe no-referencia, comparar con todas las referencias
                for shot_id in shots:
                    # Saltar si el golpe ya es una referencia
                    if shot_id in refs:
                        continue
                    
                    shot_meta = self.shot_metadata.get(shot_id, {})
                    
                    for ref_id in refs:
                        ref_meta = self.shot_metadata.get(ref_id, {})
                        
                        # Calcular distancia DTW
                        distance, _ = self.compare_shots(shot_id, ref_id)
                        
                        # Ignorar comparaciones inválidas
                        if np.isinf(distance):
                            continue
                        
                        # Guardar resultados
                        comparisons.append({
                            'shot_id': shot_id,
                            'player': shot_meta.get('player', 'Unknown'),
                            'reference_id': ref_id,
                            'reference_player': ref_meta.get('player', 'Unknown'),
                            'stroke_type': stroke_type,
                            'spin_type': spin_type,
                            'dtw_distance': distance
                        })
            else:
                logging.warning(f"No hay referencias para {stroke_type} {spin_type}")
        
        if not comparisons:
            logging.warning("No se pudieron realizar comparaciones entre golpes y referencias")
            return None
        
        # Crear DataFrame
        results_df = pd.DataFrame(comparisons)
        
        # Ordenar por tipo de golpe, tipo de efecto y distancia
        results_df = results_df.sort_values(by=['stroke_type', 'spin_type', 'dtw_distance'])
        
        # Guardar a CSV si se especificó un archivo
        if output_file:
            results_df.to_csv(output_file, index=False)
            logging.info(f"Informe guardado en {output_file}")
        
        return results_df

    def create_category_comparison_report(self, output_file=None, filter_by_stroke=None, filter_by_spin=None) -> pd.DataFrame:
        """
        Genera un informe detallado de comparación entre golpes de la misma categoría.
        
        Args:
            output_file: Nombre del archivo CSV para guardar el informe
            filter_by_stroke: Filtrar solo ciertos tipos de golpe (None = todos)
            filter_by_spin: Filtrar solo ciertos tipos de efecto (None = todos)
                
        Returns:
            DataFrame con resultados de comparación
        """
        # Asegurarse de que tengamos datos cargados
        if len(self.keypoints_data) == 0:
            logging.error("No hay datos de keypoints cargados. Carga primero algunos golpes.")
            return None
        
        # Organizar golpes por categoría (stroke_type + spin_type)
        categories = {}
        
        for shot_id, meta in self.shot_metadata.items():
            stroke_type = meta.get('stroke_type')
            spin_type = meta.get('spin_type')
            
            # Aplicar filtros si existen
            if filter_by_stroke and stroke_type != filter_by_stroke:
                continue
            if filter_by_spin and spin_type != filter_by_spin:
                continue
            
            # Crear clave de categoría
            if not stroke_type or not spin_type:
                continue  # Ignorar si falta información
                
            category_key = f"{stroke_type}/{spin_type}"
            
            # Organizar por carpeta para asegurar comparación solo dentro de la misma estructura
            folder = meta.get('folder', '')
            
            if category_key not in categories:
                categories[category_key] = {}
            
            if folder not in categories[category_key]:
                categories[category_key][folder] = []
                
            categories[category_key][folder].append(shot_id)
        
        # Realizar comparaciones dentro de cada categoría y carpeta
        comparison_data = []
        
        for category_key, folders in categories.items():
            for folder, shots in folders.items():
                n_shots = len(shots)
                
                # Solo comparar si hay al menos 2 golpes en la categoría
                if n_shots < 2:
                    continue
                    
                logging.info(f"Comparando {n_shots} golpes en categoría {category_key}, carpeta {folder}")
                
                # Comparar cada par de golpes (solo la mitad superior de la matriz)
                for i in range(n_shots):
                    shot_id1 = shots[i]
                    meta1 = self.shot_metadata.get(shot_id1, {})
                    
                    for j in range(i+1, n_shots):
                        shot_id2 = shots[j]
                        meta2 = self.shot_metadata.get(shot_id2, {})
                        
                        # Calcular distancia DTW
                        distance = self.get_or_calculate_distance(shot_id1, shot_id2)
                        
                        # Ignorar comparaciones inválidas
                        if np.isinf(distance):
                            continue
                        
                        # Guardar resultados
                        comparison_data.append({
                            'category': category_key,
                            'folder': folder,
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
            logging.warning("No se pudieron realizar comparaciones entre golpes")
            return None
                
        # Crear DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Guardar a CSV si se especificó un archivo
        if output_file:
            df.to_csv(output_file, index=False)
            logging.info(f"Informe guardado en {output_file}")
        
        return df
    
    def benchmark_against_reference(self, input_shot_id):
        """
        Compara un golpe contra los golpes de referencia apropiados (misma categoría).
        
        Args:
            input_shot_id: ID del golpe a comparar
            
        Returns:
            DataFrame con resultados de benchmark
        """
        if input_shot_id not in self.keypoints_data:
            logging.error(f"El golpe {input_shot_id} no está cargado")
            return None
        
        input_meta = self.shot_metadata.get(input_shot_id, {})
        
        if not input_meta:
            logging.error(f"No hay metadata disponible para {input_shot_id}")
            return None
        
        # Encontrar referencias del mismo tipo de golpe y efecto
        stroke_type = input_meta.get('stroke_type')
        spin_type = input_meta.get('spin_type')
        folder = input_meta.get('folder', '')
        
        if not stroke_type or not spin_type:
            logging.error(f"Tipo de golpe o efecto no disponible para {input_shot_id}")
            return None
        
        # Cargar golpes de referencia si aún no están cargados
        if not self.reference_data:
            self.load_reference_file()
            self.load_all_reference_shots()
        
        # Buscar referencias adecuadas (mismo tipo, mismo efecto, misma carpeta)
        reference_shots = []
        
        for shot_id, meta in self.shot_metadata.items():
            # No comparar consigo mismo
            if shot_id == input_shot_id:
                continue
                
            # Verificar que sea del mismo tipo y efecto
            if (meta.get('stroke_type') == stroke_type and 
                meta.get('spin_type') == spin_type and 
                meta.get('folder', '') == folder):
                reference_shots.append(shot_id)
        
        if not reference_shots:
            logging.error(f"No se encontraron golpes de referencia para {stroke_type} {spin_type} en la misma carpeta")
            return None
        
        # Comparar con cada referencia
        results = []
        
        for ref_id in reference_shots:
            distance = self.get_or_calculate_distance(input_shot_id, ref_id)
            
            if np.isinf(distance):
                continue  # Ignorar comparaciones inválidas
                
            meta = self.shot_metadata.get(ref_id, {})
            results.append({
                'reference_shot': ref_id,
                'reference_player': meta.get('player', 'Unknown'),
                'dtw_distance': distance,
                'stroke_type': stroke_type,
                'spin_type': spin_type
            })
        
        if not results:
            logging.error("No se pudieron realizar comparaciones válidas")
            return None
            
        # Ordenar por distancia
        results_df = pd.DataFrame(results).sort_values(by='dtw_distance')
        
        # Visualizar resultados
        plt.figure(figsize=(10, 6))
        
        if len(results_df) > 10:
            # Si hay muchos resultados, mostrar solo los 10 más similares
            plot_df = results_df.head(10)
        else:
            plot_df = results_df
            
        sns.barplot(x='dtw_distance', y='reference_player', data=plot_df)
        
        # Obtener información del golpe de entrada
        input_player = input_meta.get('player', 'Unknown')
        plt.title(f"Comparación de {input_player} - {stroke_type} {spin_type} con golpes de referencia")
        plt.xlabel('Distancia DTW (menor = más similar)')
        plt.ylabel('Jugador de referencia')
        plt.tight_layout()
        plt.show()
        
        return results_df

    def visualize_category_comparison(self, comparison_df, n_best=3, n_worst=3):
        """
        Visualiza los resultados de comparación entre golpes de la misma categoría.
        
        Args:
            comparison_df: DataFrame con resultados de comparación
            n_best: Número de pares más similares a visualizar
            n_worst: Número de pares menos similares a visualizar
        """
        if comparison_df is None or len(comparison_df) == 0:
            logging.error("No hay datos de comparación para visualizar")
            return
        
        # Ordenar por distancia
        sorted_df = comparison_df.sort_values(by='dtw_distance')
        
        # Extraer los más similares y menos similares
        most_similar = sorted_df.head(n_best)
        least_similar = sorted_df.tail(n_worst)
        
        # Crear visualización
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Gráfico de los más similares
        bars_most = sns.barplot(x='dtw_distance', y='category', data=most_similar, ax=axes[0])
        axes[0].set_title(f'Los {n_best} pares más similares')
        axes[0].set_xlabel('Distancia DTW (menor = más similar)')
        axes[0].set_ylabel('Categoría')
        
        # Añadir etiquetas para los más similares
        for i, row in enumerate(most_similar.itertuples()):
            player1 = row.player1.split()[0] if isinstance(row.player1, str) else ''
            player2 = row.player2.split()[0] if isinstance(row.player2, str) else ''
            label = f"{player1} vs {player2}"
            axes[0].text(row.dtw_distance, i, label, va='center', fontsize=8)
        
        # Gráfico de los menos similares
        bars_least = sns.barplot(x='dtw_distance', y='category', data=least_similar, ax=axes[1])
        axes[1].set_title(f'Los {n_worst} pares menos similares')
        axes[1].set_xlabel('Distancia DTW (mayor = menos similar)')
        axes[1].set_ylabel('Categoría')
        
        # Añadir etiquetas para los menos similares
        for i, row in enumerate(least_similar.itertuples()):
            player1 = row.player1.split()[0] if isinstance(row.player1, str) else ''
            player2 = row.player2.split()[0] if isinstance(row.player2, str) else ''
            label = f"{player1} vs {player2}"
            axes[1].text(row.dtw_distance, i, label, va='center', fontsize=8)
        
        plt.tight_layout()
        plt.show()

    def create_reference_shots_report(self, output_file=None):
        """
        Genera un informe de comparación entre todos los golpes de referencia.
        
        Args:
            output_file: Nombre del archivo CSV para guardar el informe
                
        Returns:
            DataFrame con resultados de comparación
        """
        if not self.reference_data:
            logging.error("No hay datos de referencia cargados. Primero ejecuta load_reference_file().")
            return None
        
        # Asegurarse de que todos los golpes de referencia estén cargados
        self.load_all_reference_shots()
        
        # Extraer todos los IDs de golpes de referencia
        reference_shot_ids = []
        for stroke_type, spins in self.reference_data.items():
            for spin_type, shot_ids in spins.items():  # shot_ids es ahora una lista
                for shot_id in shot_ids:  # Iterar directamente sobre los shot_ids (lista)
                    if shot_id in self.keypoints_data:
                        reference_shot_ids.append((shot_id, stroke_type, spin_type))
        
        # Crear matriz de comparación
        comparison_data = []
        
        for i, (shot_id1, stroke_type1, spin_type1) in enumerate(reference_shot_ids):
            meta1 = self.shot_metadata.get(shot_id1, {})
            folder1 = meta1.get('folder', '')
            
            for j, (shot_id2, stroke_type2, spin_type2) in enumerate(reference_shot_ids):
                if i >= j:  # Solo la mitad superior de la matriz
                    continue

                # **Filtrar solo golpes con el mismo stroke_type y spin_type**
                if stroke_type1 != stroke_type2 or spin_type1 != spin_type2:
                    continue  # Saltar si los tipos no coinciden
                
                meta2 = self.shot_metadata.get(shot_id2, {})
                folder2 = meta2.get('folder', '')
                
                # Verificar que estén en la misma carpeta
                if folder1 != folder2:
                    continue  # Saltar si no están en la misma carpeta
                
                # Calcular distancia DTW
                distance, _ = self.compare_shots(shot_id1, shot_id2)
                
                # Ignorar comparaciones inválidas
                if np.isinf(distance):
                    continue
                
                # Guardar resultados
                comparison_data.append({
                    'shot_id1': shot_id1,
                    'player1': meta1.get('player', 'Unknown'),
                    'stroke_type1': stroke_type1,
                    'spin_type1': spin_type1,
                    'shot_id2': shot_id2,
                    'player2': meta2.get('player', 'Unknown'),
                    'stroke_type2': stroke_type2,
                    'spin_type2': spin_type2,
                    'dtw_distance': distance,
                    'folder': folder1
                })
        
        if not comparison_data:
            logging.warning("No se pudieron realizar comparaciones entre golpes de referencia")
            return None
            
        # Crear DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Guardar a CSV si se especificó un archivo
        if output_file:
            df.to_csv(output_file, index=False)
            logging.info(f"Informe guardado en {output_file}")
        
        return df

    def run_full_analysis(self, output_folder="reports"):
        """
        Ejecuta el análisis completo del sistema de comparación de golpes.
        
        Args:
            output_folder: Carpeta donde guardar los informes
        """
        # Crear carpeta de salida si no existe
        output_dir = Path(output_folder)
        output_dir.mkdir(exist_ok=True)
        
        # 1. Cargar archivo de referencia
        self.load_reference_file()
        
        # 2. Cargar todos los golpes disponibles
        self.load_all_available_shots()
        
        # 3. Generar informe global por categorías
        logging.info("Generando informe de comparación por categorías...")
        comparison_df = self.create_category_comparison_report(
            output_file=output_dir / "category_comparison.csv"
        )
        
        # 4. Visualizar resultados generales
        if comparison_df is not None:
            self.visualize_category_comparison(comparison_df)
        
        # 5. Generar informes específicos por tipo de golpe
        stroke_types = set()
        for _, meta in self.shot_metadata.items():
            stroke_type = meta.get('stroke_type')
            if stroke_type:
                stroke_types.add(stroke_type)
        
        for stroke in sorted(stroke_types):
            logging.info(f"Generando informe para golpes de tipo {stroke}...")
            stroke_df = self.create_category_comparison_report(
                output_file=output_dir / f"{stroke.lower().replace(' ', '_')}_comparison.csv",
                filter_by_stroke=stroke
            )
            
            # Visualizar si hay datos
            if stroke_df is not None and len(stroke_df) > 0:
                plt.figure(figsize=(10, 8))
                plt.title(f"Distribución de distancias DTW para {stroke}")
                sns.histplot(stroke_df['dtw_distance'], kde=True)
                plt.savefig(output_dir / f"{stroke.lower().replace(' ', '_')}_distribution.png")
                plt.close()
                
                # Realizar clustering
                self.cluster_shots_by_category(stroke_type=stroke, n_clusters=3)
        
        # 6. Generar informe de referencia
        logging.info("Generando informe de golpes de referencia...")
        ref_df = self.create_reference_shots_report(
            output_file=output_dir / "reference_shots_report.csv"
        )
        
        logging.info(f"Análisis completo guardado en {output_dir}")
        
        # 7. Resumen final
        summary = {
            'num_shots': len(self.keypoints_data),
            'stroke_types': list(stroke_types),
            'num_comparisons': len(comparison_df) if comparison_df is not None else 0,
            'output_folder': str(output_dir)
        }
        
        with open(output_dir / "analysis_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
            
        return summary

def main():
    """
    Función principal que compara todos los golpes disponibles contra
    los golpes de referencia definidos en reference_all.json.
    """
    # Crear el comparador
    comparator = TennisMotionComparator()
    
    output_folder = "/Users/emilio/Documents/Winner Way/scripts/dtw/reports"  # Carpeta para guardar resultados
    
    # Crear carpeta de salida si no existe
    output_dir = Path(output_folder)
    output_dir.mkdir(exist_ok=True)
    
    # Cargar archivo de referencia
    comparator.load_reference_file("reference_all.json")
    
    # Comparar todos los golpes contra los de referencia
    logging.info("Comparando todos los golpes contra los golpes de referencia...")
    results_df = comparator.compare_all_against_references(
        output_file=None  # Don't save yet - we'll save after renaming columns
    )
    
    if results_df is not None:
        # Store values for summary before renaming
        num_shots = len(results_df['shot_id'].unique())
        num_refs = len(results_df['reference_id'].unique())
        num_comparisons = len(results_df)
        
        # Store category counts before renaming
        category_counts = results_df.groupby(['stroke_type', 'spin_type']).size()
        
        # Create visualization before renaming
        plt.figure(figsize=(12, 8))
        plt.title("Distribución de distancias DTW por categoría")
        sns.boxplot(x='stroke_type', y='dtw_distance', hue='spin_type', data=results_df)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "distance_distribution.png")
        
        # Prepare best matches before renaming
        best_matches = results_df.loc[results_df.groupby(['shot_id', 'stroke_type', 'spin_type'])['dtw_distance'].idxmin()]
        best_matches = best_matches.sort_values(by=['stroke_type', 'spin_type', 'dtw_distance'])
        
        # Now rename columns on the original DataFrame
        column_mapping = {
            'shot_id': 'shot_id1',
            'player': 'player1',
            'reference_id': 'shot_id2',
            'reference_player': 'player2',
            'stroke_type': 'stroke_type1',
            'spin_type': 'spin_type1'
        }
        
        results_df = results_df.rename(columns=column_mapping)
        results_df['stroke_type2'] = results_df['stroke_type1']
        results_df['spin_type2'] = results_df['spin_type1']
        
        # Reorder columns to put dtw_distance at the end
        column_order = [
            'shot_id1', 'player1', 'shot_id2', 'player2', 
            'stroke_type1', 'spin_type1', 
            'stroke_type2', 'spin_type2', 
            'dtw_distance'
        ]
        column_order = [
            'shot_id1', 'player1', 'stroke_type1', 'spin_type1', 
            'shot_id2', 'player2', 'stroke_type2', 'spin_type2', 
            'dtw_distance'
        ]
        results_df = results_df[column_order]
        
        # Save the renamed DataFrame
        results_df.to_csv(output_dir / "all_vs_references.csv", index=False)
        
        # Rename columns for best_matches too
        best_matches = best_matches.rename(columns=column_mapping)
        best_matches['stroke_type2'] = best_matches['stroke_type1']
        best_matches['spin_type2'] = best_matches['spin_type1']
        
        # Also reorder columns in best_matches
        best_matches = best_matches[column_order]
        
        best_matches.to_csv(output_dir / "best_matches.csv", index=False)
        
        # Print summary
        logging.info(f"\nResumen de comparaciones:")
        logging.info(f"  - Total de golpes analizados: {num_shots}")
        logging.info(f"  - Total de golpes de referencia: {num_refs}")
        logging.info(f"  - Total de comparaciones realizadas: {num_comparisons}")
        
        logging.info("\nNúmero de comparaciones por categoría:")
        for (stroke, spin), count in category_counts.items():
            logging.info(f"  - {stroke} {spin}: {count} comparaciones")
        
        logging.info(f"\nLos mejores matches por golpe se han guardado en {output_dir}/best_matches.csv")
    
    logging.info(f"Análisis completado. Resultados guardados en {output_dir}")
    
    return results_df

if __name__ == "__main__":
    main()