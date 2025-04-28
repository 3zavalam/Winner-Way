import cv2
import tensorflow as tf
import numpy as np

from backend.utils.draw import draw_edges, draw_keypoints, draw_roi

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