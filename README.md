# 🤖 Clasificador de Lengua de Señas Colombiana (LSC) con CNN y TensorFlow Lite

Este proyecto implementa una red neuronal convolucional (CNN) para reconocer gestos de la Lengua de Señas Colombiana (LSC) en tiempo real utilizando TensorFlow, MediaPipe y OpenCV. El modelo se entrena con imágenes de manos etiquetadas y se convierte a TensorFlow Lite para su uso eficiente en dispositivos con recursos limitados.

## 📁 Estructura del Proyecto
```
manos-conectadas/
├── data/
│ ├── train/ # Imágenes de entrenamiento organizadas por clase
│ │ └── val/ # Imágenes de validación organizadas por clase
├── cnn/
│ ├── train_cnn.py # Entrenamiento y conversión del modelo
│ ├── prepare_dataset.py # Preparar set de datos
│ ├── main_cnn_inference.py # Inferencia en tiempo real con webcam
│ ├── convert_to_tflite.py # Convierte el modelo .h5 a .flite
├── hand_signs_cnn.h5 # Modelo entrenado en formato Keras
├── hand_signs.tflite # Modelo convertido a TensorFlow Lite
└── README.md # Documentación del proyecto
```

## 🧠 Descripción de los Scripts

### `train_cnn.py`

- **Carga y preprocesamiento de datos**: Utiliza `ImageDataGenerator` para aplicar aumentos de datos y normalización.
- **Arquitectura del modelo**: Basada en `MobileNetV2` preentrenada, seguida de capas densas para clasificación.
- **Entrenamiento**: Compila y entrena el modelo con `categorical_crossentropy` y el optimizador `Adam`.
- **Guardado del modelo**: Almacena el modelo entrenado en formato `.h5`.
- **Conversión a TFLite**: Convierte el modelo a formato `.tflite` para despliegue en dispositivos móviles o embebidos.

### `main_cnn_inference.py`

- **Captura de video**: Utiliza OpenCV para capturar video en tiempo real desde la webcam.
- **Detección de manos**: Emplea MediaPipe para detectar y extraer la región de interés (ROI) de la mano.
- **Preprocesamiento**: Redimensiona y normaliza la ROI para que coincida con la entrada esperada por el modelo.
- **Inferencia**: Carga el modelo `.tflite` y realiza predicciones en tiempo real.
- **Visualización**: Muestra la clase predicha sobre el video en tiempo real.

## 🚀 Requisitos

- Python 3.11.4 
- TensorFlow
- OpenCV
- MediaPipe
- NumPy

Instalación de dependencias:

```bash
pip install tensorflow opencv-python mediapipe numpy
```

## 🏁 Ejecución

```
python prepare_dataset.py
python train_cnn.py
python convert_to_tflite.py
python main_cnn_inference.py
```

## Contribuciones
Las contribuciones son bienvenidas. Para colaborar:
1. Crea un fork del repositorio.
2. Crea una nueva rama (`git checkout -b feature/nueva-funcionalidad`).
3. Realiza cambios y súbelos (`git commit -m "Descripción del cambio"`).
4. Envía un Pull Request.
