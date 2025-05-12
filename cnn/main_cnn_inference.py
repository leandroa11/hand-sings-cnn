import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os

# Parámetros
IMAGE_SIZE = (224, 224)  # (height, width)
CLASS_NAMES = sorted(os.listdir("data/train"))  # mapeo índices → etiquetas

# Cargar intérprete TFLite
interpreter = tf.lite.Interpreter("hand_signs.tflite")
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
input_shape    = input_details["shape"]        # e.g. [1,224,224,3]

# Para asegurar consistencia, extraemos directamente ancho y alto esperados
_, expected_h, expected_w, _ = input_shape

# MediaPipe Hands para detección de ROI
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res   = hands.process(rgb)

    if res.multi_hand_landmarks:
        # Tomamos la primera mano detectada
        lm = res.multi_hand_landmarks[0].landmark
        h, w, _ = frame.shape

        # Coordenadas absolutas de todos los landmarks
        xs = [int(p.x * w) for p in lm]
        ys = [int(p.y * h) for p in lm]
        xmin, xmax = max(min(xs) - 20, 0), min(max(xs) + 20, w)
        ymin, ymax = max(min(ys) - 20, 0), min(max(ys) + 20, h)

        roi = frame[ymin:ymax, xmin:xmax]
        # Comprobar si la ROI es válida
        try:
            roi_resized = cv2.resize(roi, (expected_w, expected_h))
        except Exception as e:
            print(f"Error al redimensionar ROI: tamaño original {roi.shape} → esperado {(expected_h, expected_w)}")
            raise

        # Preprocesar para TFLite: RGB, normalizar y añadir batch
        roi_input = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
        roi_input = roi_input.astype(np.float32) / 255.0
        roi_input = np.expand_dims(roi_input, axis=0)  # forma [1, H, W, 3]

        # Comprobación de forma
        if roi_input.shape != tuple(input_shape):
            raise ValueError(f"Input shape mismatch: got {roi_input.shape}, "
                             f"expected {tuple(input_shape)}")

        # Inferencia
        interpreter.set_tensor(input_details["index"], roi_input)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details["index"])[0]
        label = CLASS_NAMES[np.argmax(preds)]

        # Mostrar resultado
        cv2.putText(frame, label, (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,0), 2, cv2.LINE_AA)
        mp_drawing.draw_landmarks(frame,
                                  res.multi_hand_landmarks[0],
                                  mp_hands.HAND_CONNECTIONS)

    cv2.imshow("LSC CNN Inference", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
