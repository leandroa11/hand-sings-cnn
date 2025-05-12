# Importación de librerías necesarias
import cv2  # Biblioteca para procesamiento de imágenes y video
import numpy as np  # Biblioteca para operaciones matemáticas y manipulación de matrices
import mediapipe as mp  # Biblioteca para detección de manos y landmarks
import tensorflow as tf  # Biblioteca para cargar e inferir modelos de aprendizaje profundo
import os  # Biblioteca para interactuar con el sistema de archivos

# Parámetros iniciales
IMAGE_SIZE = (224, 224)  # Tamaño esperado de las imágenes de entrada (alto, ancho)
CLASS_NAMES = sorted(os.listdir("data/train"))  # Lista de clases (etiquetas) obtenidas del directorio de entrenamiento

# Cargar el modelo TFLite
interpreter = tf.lite.Interpreter("hand_signs.tflite")  # Carga el modelo preentrenado en formato TFLite
interpreter.allocate_tensors()  # Asigna memoria para los tensores del modelo
input_details = interpreter.get_input_details()[0]  # Detalles de la entrada del modelo
output_details = interpreter.get_output_details()[0]  # Detalles de la salida del modelo
input_shape = input_details["shape"]  # Forma esperada de la entrada, e.g., [1, 224, 224, 3]

# Extraer dimensiones esperadas de la entrada del modelo
_, expected_h, expected_w, _ = input_shape  # Altura y ancho esperados por el modelo

# Configuración de MediaPipe Hands para detección de manos
mp_hands = mp.solutions.hands  # Solución de MediaPipe para detección de manos
mp_drawing = mp.solutions.drawing_utils  # Utilidad para dibujar landmarks y conexiones
hands = mp_hands.Hands(
    static_image_mode=False,  # Procesar video en tiempo real
    max_num_hands=1,  # Detectar una sola mano
    min_detection_confidence=0.5,  # Confianza mínima para detectar una mano
    min_tracking_confidence=0.5  # Confianza mínima para rastrear una mano detectada
)

# Configuración de la cámara
cap = cv2.VideoCapture(0)  # Captura de video desde la cámara (índice 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Ancho del video
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Alto del video

# Bucle principal para procesar video en tiempo real
while cap.isOpened():
    ret, frame = cap.read()  # Leer un fotograma de la cámara
    if not ret:  # Si no se puede leer el fotograma, salir del bucle
        break

    frame = cv2.flip(frame, 1)  # Invertir el fotograma horizontalmente (efecto espejo)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir el fotograma de BGR a RGB
    res = hands.process(rgb)  # Procesar el fotograma con MediaPipe para detectar manos

    # Si se detectan manos en el fotograma
    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark  # Obtener landmarks de la primera mano detectada
        h, w, _ = frame.shape  # Obtener dimensiones del fotograma

        # Calcular coordenadas absolutas de los landmarks
        xs = [int(p.x * w) for p in lm]  # Coordenadas X en píxeles
        ys = [int(p.y * h) for p in lm]  # Coordenadas Y en píxeles
        xmin, xmax = max(min(xs) - 20, 0), min(max(xs) + 20, w)  # Límites horizontales de la ROI
        ymin, ymax = max(min(ys) - 20, 0), min(max(ys) + 20, h)  # Límites verticales de la ROI

        roi = frame[ymin:ymax, xmin:xmax]  # Extraer la región de interés (ROI) de la mano
        # Comprobar si la ROI es válida
        try:
            roi_resized = cv2.resize(roi, (expected_w, expected_h))  # Redimensionar la ROI al tamaño esperado
        except Exception as e:
            print(f"Error al redimensionar ROI: tamaño original {roi.shape} → esperado {(expected_h, expected_w)}")
            raise

        # Preprocesar la ROI para el modelo TFLite
        roi_input = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)  # Convertir la ROI a RGB
        roi_input = roi_input.astype(np.float32) / 255.0  # Normalizar los valores de píxeles a [0, 1]
        roi_input = np.expand_dims(roi_input, axis=0)  # Añadir una dimensión para representar el batch

        # Comprobar que la forma de la entrada coincide con la esperada por el modelo
        if roi_input.shape != tuple(input_shape):
            raise ValueError(f"Input shape mismatch: got {roi_input.shape}, "
                             f"expected {tuple(input_shape)}")

        # Realizar inferencia con el modelo
        interpreter.set_tensor(input_details["index"], roi_input)  # Establecer la entrada del modelo
        interpreter.invoke()  # Ejecutar la inferencia
        preds = interpreter.get_tensor(output_details["index"])[0]  # Obtener las predicciones del modelo
        label = CLASS_NAMES[np.argmax(preds)]  # Obtener la etiqueta de la clase con mayor probabilidad

        # Mostrar el resultado en el fotograma
        cv2.putText(frame, label, (10, 50),  # Escribir la etiqueta en el fotograma
                    cv2.FONT_HERSHEY_SIMPLEX, 1,  # Fuente y tamaño del texto
                    (0, 255, 0), 2, cv2.LINE_AA)  # Color (verde) y grosor del texto
        mp_drawing.draw_landmarks(frame,  # Dibujar landmarks y conexiones en el fotograma
                                  res.multi_hand_landmarks[0],
                                  mp_hands.HAND_CONNECTIONS)

    # Mostrar el fotograma procesado en una ventana
    cv2.imshow("LSC CNN Inference", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Salir si se presiona la tecla 'Esc'
        break

# Liberar recursos
cap.release()  # Liberar la cámara
cv2.destroyAllWindows()  # Cerrar todas las ventanas de OpenCV
