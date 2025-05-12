import os
import cv2
import mediapipe as mp

# Configuración MediaPipe Hands para detección de ROI
# Se inicializa el modelo de MediaPipe para detectar manos en imágenes.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,  # Modo dinámico para imágenes en movimiento.
                       max_num_hands=1,         # Detectar solo una mano por imagen.
                       min_detection_confidence=0.5)  # Confianza mínima para considerar una detección válida.

def extract_and_save(input_dir, output_dir, image_size=(250,250)):
    """
    Recorre las subcarpetas de input_dir, detecta la mano en cada imagen
    y guarda el recorte en output_dir en la misma estructura de subcarpetas.
    
    Args:
        input_dir (str): Directorio de entrada con imágenes organizadas por clases.
        output_dir (str): Directorio de salida donde se guardarán las imágenes procesadas.
        image_size (tuple): Tamaño al que se redimensionarán las imágenes recortadas.
    """
    # Iterar sobre las subcarpetas (clases) en el directorio de entrada.
    for label in os.listdir(input_dir):
        in_label_dir  = os.path.join(input_dir, label)  # Ruta de la subcarpeta de entrada.
        out_label_dir = os.path.join(output_dir, label)  # Ruta de la subcarpeta de salida.
        os.makedirs(out_label_dir, exist_ok=True)  # Crear la subcarpeta de salida si no existe.

        # Iterar sobre los archivos de la subcarpeta actual.
        for fname in os.listdir(in_label_dir):
            img_path = os.path.join(in_label_dir, fname)  # Ruta completa de la imagen.
            img = cv2.imread(img_path)  # Leer la imagen con OpenCV.
            if img is None:
                # Si la imagen no se puede cargar (archivo corrupto o no válido), se omite.
                continue

            # Convertir la imagen de BGR (formato OpenCV) a RGB (formato requerido por MediaPipe).
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Procesar la imagen con MediaPipe para detectar landmarks de la mano.
            res = hands.process(rgb)
            if not res.multi_hand_landmarks:
                # Si no se detecta ninguna mano en la imagen, se omite.
                continue

            # Obtener las dimensiones de la imagen.
            h, w, _ = img.shape
            # Calcular las coordenadas de los landmarks escaladas a las dimensiones de la imagen.
            x_coords = [lm.x * w for lm in res.multi_hand_landmarks[0].landmark]
            y_coords = [lm.y * h for lm in res.multi_hand_landmarks[0].landmark]
            # Determinar el rectángulo mínimo que contiene todos los landmarks (bounding box).
            xmin, xmax = int(min(x_coords)), int(max(x_coords))
            ymin, ymax = int(min(y_coords)), int(max(y_coords))

            # Añadir un margen al bounding box para incluir un área adicional alrededor de la mano.
            pad = 20
            xmin, ymin = max(0, xmin-pad), max(0, ymin-pad)  # Asegurarse de no salir de los límites de la imagen.
            xmax, ymax = min(w, xmax+pad), min(h, ymax+pad)
            # Recortar la región de interés (ROI) de la imagen original.
            roi = img[ymin:ymax, xmin:xmax]

            # Redimensionar la ROI al tamaño especificado para que sea compatible con la CNN.
            roi_resized = cv2.resize(roi, image_size)
            # Construir la ruta de salida para guardar la imagen procesada.
            out_path = os.path.join(out_label_dir, fname)
            # Guardar la imagen procesada en el directorio de salida.
            cv2.imwrite(out_path, roi_resized)

if __name__ == "__main__":
    # Ajusta estas rutas según tu estructura de directorios.
    # Procesar las imágenes de entrenamiento.
    extract_and_save("data/train", "data/train_output")
    # Procesar las imágenes de validación.
    extract_and_save("data/validation", "data/val_output")