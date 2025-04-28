import json
import cv2
import tensorflow as tf
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import os

class RoI:
    """ Define la Región de Interés (RoI) alrededor del jugador de tenis. """
    def __init__(self, shape):
        self.frame_width = shape[1]
        self.frame_height = shape[0]
        self.width = self.frame_width
        self.height = self.frame_height
        self.center_x = shape[1] // 2
        self.center_y = shape[0] // 2
        self.valid = False

    def extract_subframe(self, frame):
        """Extrae la RoI del frame original"""
        y_start = max(0, self.center_y - self.height // 2)
        y_end = min(self.frame_height, self.center_y + self.height // 2)
        x_start = max(0, self.center_x - self.width // 2)
        x_end = min(self.frame_width, self.center_x + self.width // 2)
        
        return frame[y_start:y_end, x_start:x_end].copy()

    def transform_to_subframe_coordinates(self, keypoints_from_tf):
        """Transforma las coordenadas de los keypoints a coordenadas de subframe"""
        # Movenet usa valores normalizados entre 0 y 1, multiplicamos por las dimensiones
        keypoints = np.squeeze(keypoints_from_tf)
        subframe_keypoints = keypoints.copy()
        # y, x, confidence
        subframe_keypoints[:, 0] *= self.height
        subframe_keypoints[:, 1] *= self.width
        return subframe_keypoints

    def transform_to_frame_coordinates(self, keypoints_from_tf):
        """Transforma las coordenadas de los keypoints a coordenadas del frame completo"""
        keypoints_pixels_subframe = self.transform_to_subframe_coordinates(keypoints_from_tf)
        keypoints_pixels_frame = keypoints_pixels_subframe.copy()
        
        # Añadir offset para convertir de coordenadas del subframe a coordenadas del frame completo
        y_offset = max(0, self.center_y - self.height // 2)
        x_offset = max(0, self.center_x - self.width // 2)
        
        # Añadir los offsets a las coordenadas correspondientes (y, x)
        keypoints_pixels_frame[:, 0] += y_offset
        keypoints_pixels_frame[:, 1] += x_offset
        
        return keypoints_pixels_frame

    def update(self, keypoints_pixels):
        """Actualiza la RoI con nuevos keypoints"""
        # Filtrar keypoints con confianza > 0.1
        valid_keypoints = keypoints_pixels[keypoints_pixels[:, 2] > 0.1]
        if len(valid_keypoints) < 5:  # Necesitamos suficientes puntos válidos
            self.reset()
            return

        min_x = int(min(valid_keypoints[:, 1]))
        min_y = int(min(valid_keypoints[:, 0]))
        max_x = int(max(valid_keypoints[:, 1]))
        max_y = int(max(valid_keypoints[:, 0]))

        self.center_x = (min_x + max_x) // 2
        self.center_y = (min_y + max_y) // 2

        prob_mean = np.mean(valid_keypoints[:, 2])
        if self.width != self.frame_width and prob_mean < 0.3:
            self.reset()
            return

        self.width = int((max_x - min_x) * 1.3)
        self.height = int((max_y - min_y) * 1.3)

        if self.height < 150 or self.width < 10:
            self.reset()
            return

        self.width = max(self.width, self.height)
        self.height = max(self.width, self.height)

        # Asegurar que la ROI está dentro de los límites del frame
        if self.center_x + self.width // 2 >= self.frame_width:
            self.center_x = self.frame_width - self.width // 2 - 1

        if 0 > self.center_x - self.width // 2:
            self.center_x = self.width // 2 + 1

        if self.center_y + self.height // 2 >= self.frame_height:
            self.center_y = self.frame_height - self.height // 2 - 2

        if 0 > self.center_y - self.height // 2:
            self.center_y = self.height // 2 + 1

        self.valid = True

    def reset(self):
        """Reinicia la RoI al tamaño completo de la imagen"""
        self.width = self.frame_width
        self.height = self.frame_height
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2
        self.valid = False

    def draw_shot(self, frame, stroke_type, shot_variant=None):
        """Dibuja la información del tiro sobre la caja de la RoI"""
        shot_text = f"{stroke_type}"
        if shot_variant:
            shot_text += f" - {shot_variant}"
            
        cv2.putText(frame, shot_text, 
                   (self.center_x - self.width // 4, self.center_y - self.height // 2 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(128, 255, 255), thickness=2)


class HumanPoseExtractor:
    """Define el mapeo entre los keypoints de movenet y las partes del cuerpo"""
    EDGES = {
        (0, 1): "m", (0, 2): "c", (1, 3): "m", (2, 4): "c", (0, 5): "m", (0, 6): "c",
        (5, 7): "m", (7, 9): "m", (6, 8): "c", (8, 10): "c", (5, 6): "y", (5, 11): "m",
        (6, 12): "c", (11, 12): "y", (11, 13): "m", (13, 15): "m", (12, 14): "c", (14, 16): "c",
    }

    COLORS = {"c": (255, 255, 0), "m": (255, 0, 255), "y": (0, 255, 255)}

    KEYPOINT_DICT = {
        "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
        "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
        "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12, "left_knee": 13,
        "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
    }

    def __init__(self, shape, model_path="backend/models/movenet.tflite"):
        # Initialize the TFLite interpreter
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.roi = RoI(shape)

    def extract(self, frame):
        """Run inference model on subframe"""
        subframe = self.roi.extract_subframe(frame)
        
        # Asegurar que subframe no esté vacío
        if subframe.size == 0:
            print("Warning: ROI extraction resulted in empty subframe. Resetting ROI.")
            self.roi.reset()
            subframe = self.roi.extract_subframe(frame)
        
        # Resize to match model input size
        img = tf.image.resize_with_pad(np.expand_dims(subframe, axis=0), 192, 192)
        input_image = tf.cast(img, dtype=tf.uint8)

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.interpreter.set_tensor(input_details[0]["index"], np.array(input_image))
        self.interpreter.invoke()
        self.keypoints_with_scores = self.interpreter.get_tensor(output_details[0]["index"])
        self.keypoints_pixels_frame = self.roi.transform_to_frame_coordinates(self.keypoints_with_scores)

    def discard(self, list_of_keypoints):
        """Discard some points like eyes or ears (useless for our application)"""
        for keypoint in list_of_keypoints:
            self.keypoints_with_scores[0, 0, self.KEYPOINT_DICT[keypoint], 2] = 0
            self.keypoints_pixels_frame[self.KEYPOINT_DICT[keypoint], 2] = 0

    def draw_results_subframe(self, frame):
        """Draw key points and edges on subframe (RoI)"""
        subframe = self.roi.extract_subframe(frame)
        keypoints_pixels_subframe = self.roi.transform_to_subframe_coordinates(self.keypoints_with_scores)
        draw_edges(subframe, keypoints_pixels_subframe, self.EDGES, 0.2)
        draw_keypoints(subframe, keypoints_pixels_subframe, 0.2)
        return subframe

    def draw_results_frame(self, frame):
        """Draw key points and edges on frame"""
        if not self.roi.valid:
            return
        draw_edges(frame, self.keypoints_pixels_frame, self.EDGES, 0.01)
        draw_keypoints(frame, self.keypoints_pixels_frame, 0.01)
        draw_roi(self.roi, frame)

    def get_keypoints_list(self):
        """Devuelve la lista de keypoints en un formato más amigable para JSON"""
        keypoints_list = []
        for i, keypoint_name in enumerate(self.KEYPOINT_DICT.keys()):
            y, x, confidence = self.keypoints_pixels_frame[i]
            keypoints_list.append({
                "name": keypoint_name,
                "y": float(y),
                "x": float(x),
                "confidence": float(confidence)
            })
        return keypoints_list


def find_clip_in_metadata(metadata, video_filename):
    """
    Encuentra todos los clips en la metadata que corresponden al video dado
    """
    base_filename = os.path.basename(video_filename)
    clips = [entry for entry in metadata if entry.get('input_video') == base_filename]
    return clips


def draw_keypoints(frame, keypoints, confidence_threshold=0.1):
    """Dibuja los keypoints detectados en el frame"""
    for i, (y, x, confidence) in enumerate(keypoints):
        if confidence > confidence_threshold:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Dibujar círculo en cada keypoint

def draw_edges(frame, shaped, edges, confidence_threshold):
    """Dibuja las conexiones con cian para el lado derecho, magenta para el izquierdo y amarillo para el resto"""
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        # Comprobar que ambos puntos tengan suficiente confianza
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            # Obtener el color según el diccionario COLORS
            color_to_use = HumanPoseExtractor.COLORS.get(color, (0, 255, 255))  # Amarillo por defecto
            # Dibujar la línea entre los keypoints con el color apropiado
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color=color_to_use, thickness=2)


def draw_roi(roi, frame):
    """Dibuja la Región de Interés (RoI) con un cuadro amarillo"""
    # Dibujar las 4 líneas que forman el cuadrado de la ROI
    cv2.line(frame, 
             (roi.center_x - roi.width // 2, roi.center_y - roi.height // 2),
             (roi.center_x - roi.width // 2, roi.center_y + roi.height // 2),
             (0, 255, 255), 3)  # Línea izquierda (amarillo)

    cv2.line(frame, 
             (roi.center_x + roi.width // 2, roi.center_y + roi.height // 2),
             (roi.center_x - roi.width // 2, roi.center_y + roi.height // 2),
             (0, 255, 255), 3)  # Línea inferior (amarillo)

    cv2.line(frame, 
             (roi.center_x + roi.width // 2, roi.center_y + roi.height // 2),
             (roi.center_x + roi.width // 2, roi.center_y - roi.height // 2),
             (0, 255, 255), 3)  # Línea derecha (amarillo)

    cv2.line(frame, 
             (roi.center_x - roi.width // 2, roi.center_y - roi.height // 2),
             (roi.center_x + roi.width // 2, roi.center_y - roi.height // 2),
             (0, 255, 255), 3)  # Línea superior (amarillo)



def process_video(video_file, metadata, output_dir="json/", model_path="backend/models/movenet.tflite", visualize=True):
    """
    Procesa un video completo, extrae keypoints por frame y guarda los resultados en formato JSON.
    Dibuja los keypoints, las conexiones y el tipo de tiro.
    """
    # Crear directorio de salida si no existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Verificar que el archivo exista
    if not Path(video_file).exists():
        print(f"Error: El archivo {video_file} no existe.")
        return False

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_file}.")
        return False

    # Obtener FPS e información del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = frame_count / fps if fps > 0 else 0
    print(f"Video: {video_file}, FPS: {fps}, Frames: {frame_count}, Duración: {video_duration:.2f}s")

    # Buscar clips relevantes para este video
    base_filename = os.path.basename(video_file)
    video_clips = find_clip_in_metadata(metadata, base_filename)
    
    if not video_clips:
        print(f"No se encontraron clips definidos en la metadata para el video {base_filename}")
        return False
    
    print(f"Encontrados {len(video_clips)} clips para procesar en {base_filename}")

    # Inicializar el extractor de poses
    ret, first_frame = cap.read()
    if not ret:
        print(f"Error: No se pudo leer el primer frame del video {video_file}.")
        return False
    
    human_pose_extractor = HumanPoseExtractor(first_frame.shape, model_path)
    
    processed_clips = 0
    
    # Procesar cada clip definido en la metadata
    for clip in video_clips:
        # Volver al inicio del video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Verificar campos obligatorios
        if 'start_time_seconds' not in clip:
            print(f"Error: Falta el campo 'start_time_seconds' en un clip de {base_filename}.")
            continue
            
        if 'duration_seconds' not in clip:
            print(f"Error: Falta el campo 'duration_seconds' en un clip de {base_filename}.")
            continue
            
        if 'output_filename' not in clip:
            print(f"Error: Falta el campo 'output_filename' en un clip de {base_filename}.")
            continue
        
        start_time = clip['start_time_seconds']
        duration = clip['duration_seconds']
        end_time = start_time + duration
        
        # Calcular los frames correspondientes
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        print(f"Procesando clip: {clip['output_filename']} (Tiempo: {start_time}-{end_time}s, Frames: {start_frame}-{end_frame})")
        
        # Posicionar en el frame de inicio
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Crear la estructura de carpetas en `output_dir` usando la metadata `output_path`
        output_path = clip.get("output_path", "").replace("data/videos/", output_dir + "/")
        output_folder = Path(output_path).parent  # Obtener el directorio sin el nombre del archivo

        # Crear el directorio de salida si no existe
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Inicializar lista para los datos del clip
        clip_data = []
        frame_in_clip = 0
        current_frame = start_frame
        
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                print(f"Error: No se pudo leer el frame {current_frame}")
                break
            
            # Extraer la pose
            human_pose_extractor.extract(frame)
            human_pose_extractor.discard(["left_eye", "right_eye", "left_ear", "right_ear"])
            
            # Actualizar la RoI en función de los keypoints
            human_pose_extractor.roi.update(human_pose_extractor.keypoints_pixels_frame)
            
            # Guardar datos del frame si estamos dentro del clip
            frame_data = clip.copy()
            frame_data['frame_number'] = current_frame
            frame_data['frame_in_clip'] = frame_in_clip
            frame_data['timestamp'] = current_frame / fps
            frame_data['keypoints_raw'] = human_pose_extractor.keypoints_with_scores.tolist()
            frame_data['keypoints_list'] = human_pose_extractor.get_keypoints_list()
            frame_data['roi'] = {
                'center_x': human_pose_extractor.roi.center_x,
                'center_y': human_pose_extractor.roi.center_y,
                'width': human_pose_extractor.roi.width,
                'height': human_pose_extractor.roi.height,
                'valid': human_pose_extractor.roi.valid
            }
            
            clip_data.append(frame_data)
            
            # Visualización si está habilitada
            if visualize:
                # Dibujar los keypoints y las conexiones
                human_pose_extractor.draw_results_frame(frame)
                
                # Dibujar el tipo de golpe encima del ROI si está disponible
                if 'stroke_type' in clip:
                    shot_variant = clip.get('shot_variant', '')
                    human_pose_extractor.roi.draw_shot(frame, clip['stroke_type'], shot_variant)
                
                # Mostrar información adicional en la pantalla
                cv2.putText(frame, 
                            f"Frame: {current_frame}/{end_frame}", 
                            (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (255, 255, 255), 2)

                # Mostrar el frame
                cv2.imshow("Tennis Pose Detection", frame)
                key = cv2.waitKey(1)
                if key == 27:  # Escape para salir
                    break
            
            frame_in_clip += 1
            current_frame += 1
        
       # Guardar todos los datos del clip en un único archivo JSON con el nombre del `output_filename`
        if clip_data:
            base_filename_without_extension = Path(clip['output_filename']).stem  # Elimina la extensión
            json_filename = output_folder / f"{base_filename_without_extension}.json"  # Agrega .json
            
            with open(json_filename, "w") as f:
                json.dump(clip_data, f, indent=2)
            print(f"Datos del clip guardados en {json_filename}")

        processed_clips += 1
    
    cap.release()
    if visualize:
        cv2.destroyAllWindows()
    
    print(f"Procesamiento del video {video_file} completado. {processed_clips} clips procesados.")
    return True


def process_all_videos(metadata_file="metadata.json", output_dir="json/", video_dir="data/videos/_raw_videos/", model_path="backend/models/movenet.tflite", visualize=True):
    """
    Procesa todos los videos referenciados en el archivo de metadata
    """
    # Cargar metadata
    try:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de metadata {metadata_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: El archivo {metadata_file} no es un JSON válido")
        return
    
    # Crear un conjunto de videos únicos para procesar
    unique_videos = set(entry['input_video'] for entry in metadata if 'input_video' in entry)
    
    print(f"Se encontraron {len(unique_videos)} videos únicos para procesar.")
    
    processed_videos = 0
    failed_videos = []
    
    # Procesar cada video
    for video_name in unique_videos:
        video_path = os.path.join(video_dir, video_name)
        print(f"\n{'='*50}")
        print(f"Procesando video {processed_videos+1} de {len(unique_videos)}: {video_path}")
        print(f"{'='*50}")
        
        success = process_video(video_path, metadata, output_dir, model_path, visualize)
        
        if success:
            processed_videos += 1
        else:
            failed_videos.append(video_name)
    
    # Mostrar resumen
    print(f"\n{'='*50}")
    print(f"Procesamiento completo!")
    print(f"Videos procesados correctamente: {processed_videos} de {len(unique_videos)}")
    
    if failed_videos:
        print(f"Videos con errores ({len(failed_videos)}):")
        for vid in failed_videos:
            print(f" - {vid}")
    
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Process tennis videos to extract human poses")
    parser.add_argument("--metadata", default="metadata.json", help="Path to metadata JSON file")
    parser.add_argument("--output", default="output", help="Directory to store output files")
    parser.add_argument("--videos", default="data/videos/_raw_videos", help="Directory containing input videos")
    parser.add_argument("--model", default="backend/models/movenet.tflite", help="Path to MoveNet TFLite model")
    parser.add_argument("--no-visual", action="store_true", help="Disable visualization")
    args = parser.parse_args()
    
    # Procesar todos los videos
    process_all_videos(
        metadata_file=args.metadata, 
        output_dir=args.output, 
        video_dir=args.videos,
        model_path=args.model,
        visualize=not args.no_visual
    )